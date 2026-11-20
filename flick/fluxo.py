"""Coordenação das camadas, entrega por quadro e aplicação do delta."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

from . import blocos, track
from .imagens import (Sequencia, gravar_exr, leitor_mascara, ler_cor, nome, normalizar,
                      preparar_mascaras, recorte_nativo)
from .medidas import medir, salvar_relatorios
from .movimento import trava
from .recursos import Recursos


def gravar_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def leitores(a):
    seq = Sequencia(a.entrada, a.inicio, a.fim, a.origem_npy)
    regiao = leitor_mascara(a.cryptomatte or a.regiao, a.inicio, seq.offset,
                            a.id_crypto, a.camada_crypto)
    if bool(a.cryptomatte) != bool(a.id_crypto):
        raise ValueError("Combine --cryptomatte com pelo menos um --id-crypto.")
    return seq, regiao


def geometria(seq, a):
    h, w = seq.quadro(0).shape[:2]
    tamanho = tuple(a.tela or (w, h))
    faixa = tuple(a.faixa or (0, 0, *tamanho))
    x, y, fw, fh = faixa
    if min(x, y) < 0 or min(fw, fh) < 1 or x+fw > tamanho[0] or y+fh > tamanho[1]:
        raise ValueError("A faixa ultrapassa a tela declarada.")
    if (w, h) != tamanho and not (seq.matriz is not None and (w, h) == (fw, fh)):
        raise ValueError("A entrada deve conter a tela completa ou um NPY da faixa exata.")
    return tamanho, faixa


def preparar_entrada(seq, tamanho, faixa, pasta):
    amostra = seq.quadro(0)
    x, y, w, h = faixa
    ja_recortado = amostra.shape[:2] == (h, w)
    if (seq.matriz is not None and ja_recortado and seq.offset == 0
            and len(seq.matriz) == seq.quantidade):
        return Path(seq.padrao)
    path = pasta / "entrada.npy"
    volume = np.lib.format.open_memmap(path, "w+", amostra.dtype, (seq.quantidade, h, w, 3))
    for i in range(seq.quantidade):
        f = seq.quadro(i)
        if f.shape != amostra.shape or f.dtype != amostra.dtype:
            raise ValueError(f"Resolução ou tipo mudou no quadro {seq.primeiro+i}.")
        volume[i] = f if ja_recortado else f[y:y+h, x:x+w]
    volume.flush()
    return path


def projetar(delta, tamanho, faixa, destino):
    """Reduz só a faixa quando as bordas coincidem com pixels da entrega."""
    if min(destino) < 1:
        raise ValueError("Tamanho de saída inválido.")
    x, y, w, h = faixa
    dw, dh = destino
    tw, th = tamanho
    bordas = (x*dw/tw, y*dh/th, (x+w)*dw/tw, (y+h)*dh/th)
    if all(v.is_integer() for v in bordas):
        xa, ya, xb, yb = map(int, bordas)
        d = cv2.resize(delta, (xb-xa, yb-ya), interpolation=cv2.INTER_AREA)
        if (xa, ya, xb, yb) == (0, 0, dw, dh):
            return d
        tela = np.zeros((dh, dw, 3), np.float32)
        tela[ya:yb, xa:xb] = d
        return tela
    tela = np.zeros((th, tw, 3), np.float32)
    tela[y:y+h, x:x+w] = delta
    return cv2.resize(tela, destino, interpolation=cv2.INTER_AREA)


def exportar(camadas, saida, tamanho, faixa, destino, primeiro, precisao, indices,
             limite_fluxo=None, entrada=None, regiao=None, limiar_textura=None, teto=None):
    volumes = [np.load(p, mmap_mode="r") for p in camadas]
    if limite_fluxo is not None:
        entrada = np.load(entrada, mmap_mode="r")
        regiao = np.load(regiao, mmap_mode="r") if regiao else None
    registro = {}
    pastas = [saida/f"camada{i+1}" for i in range(len(volumes))] + [saida/"soma", saida/"acao"]
    for pasta in pastas:
        pasta.mkdir()
    for ordem, i in enumerate(indices, 1):
        q = primeiro+i
        soma = np.zeros(volumes[0].shape[1:], np.float32)
        acao = np.zeros((destino[1], destino[0]), bool)
        if limite_fluxo is not None:
            for vol in volumes:
                soma += vol[i]
            ganho, registro[q] = trava(entrada, regiao, i, soma, limite_fluxo, limiar_textura, teto)
            soma_final = np.zeros((*destino[::-1], 3), np.float32)
        for vol, pasta in zip(volumes, pastas):
            d = vol[i].astype(np.float32)
            if limite_fluxo is None:
                soma += d
            else:
                d *= ganho
            tela = projetar(d, tamanho, faixa, destino)
            if limite_fluxo is not None:
                if precisao == "half":
                    tela = tela.astype(np.float16)
                soma_final += tela
            gravar_exr(pasta/f"delta_{q:04d}.exr", tela, precisao)
            acao |= np.any(tela != 0, axis=2)
        if limite_fluxo is None:
            soma_final = projetar(soma, tamanho, faixa, destino)
        gravar_exr(pastas[-2]/f"delta_{q:04d}.exr", soma_final, precisao)
        if not cv2.imwrite(str(pastas[-1]/f"acao_{q:04d}.png"), acao.astype(np.uint8)*255):
            raise OSError("Erro ao gravar a máscara de ação.")
        if ordem % 20 == 0:
            print(f"Exportados {ordem}/{len(indices)} quadros", flush=True)
    if limite_fluxo is not None:
        gravar_json(saida/"trava_fluxo.json", {"limite_px_por_quadro": limite_fluxo,
                    "limiar_textura":limiar_textura, "teto_delta":teto, "quadros": registro})


def rodar(a, receita):
    cv2.setNumThreads(1)
    recursos = Recursos()
    seq, regiao = leitores(a)
    tamanho, faixa = geometria(seq, a)
    inicio_entrega, fim_entrega = a.entregar or (a.inicio, a.fim)
    if not a.inicio <= inicio_entrega <= fim_entrega <= a.fim:
        raise ValueError("O intervalo de entrega precisa estar contido na sequência.")
    if a.trava_fluxo is not None and (not np.isfinite(a.trava_fluxo) or a.trava_fluxo <= 0):
        raise ValueError("O limite da trava de fluxo precisa ser finito e positivo.")
    if a.trava_textura is not None and (a.trava_fluxo is None or not 0 <= a.trava_textura < 1):
        raise ValueError("A trava de textura requer fluxo e correlação no intervalo [0,1).")
    if a.teto_delta is not None and (a.trava_fluxo is None or not 0 < a.teto_delta <= 1):
        raise ValueError("O teto do delta requer a trava de fluxo e amplitude entre zero e um.")
    indices = range(inicio_entrega - a.inicio, fim_entrega - a.inicio + 1)
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=False)
    cache = pasta/"cache"
    cache.mkdir()
    gravar_json(pasta/"receita.json", receita)
    inicio = time.monotonic()
    entrada = preparar_entrada(seq, tamanho, faixa, cache)
    exclusoes = [leitor_mascara(p, a.inicio, seq.offset) for p in a.excluir]
    mask, neg = preparar_mascaras(regiao, exclusoes, tamanho, faixa, seq.quantidade,
                                  cache, receita["regiao"])
    tempos = {"preparo": time.monotonic()-inicio}
    camadas = []
    for n in range(1, 4):
        print(f"Camada {n}", flush=True)
        inicio = time.monotonic()
        dest = cache/f"camada{n}.npy"
        rel = blocos.executar(entrada, mask, camadas, dest, receita[f"camada{n}"],
                              a.processos, tamanho[0], a.inicio)
        camadas.append(dest)
        gravar_json(pasta/f"registro{n}.json", rel)
        tempos[f"camada{n}"] = time.monotonic()-inicio
    inicio = time.monotonic()
    print("Camada 4", flush=True)
    juntas = cache/"base12.npy"
    c1, c2 = [np.load(p, mmap_mode="r") for p in camadas[:2]]
    base = np.lib.format.open_memmap(juntas, "w+", np.float16, c1.shape)
    for i in range(seq.quantidade):
        base[i] = c1[i].astype(np.float32) + c2[i].astype(np.float32)
    base.flush()
    del base, c1, c2
    dest, rel = track.executar(entrada, [juntas, camadas[2]], mask, cache, receita["camada4"],
                               a.inicio, a.processos, tamanho[0]/3840, faixa[:2], neg)
    camadas.append(dest)
    gravar_json(pasta/"registro4.json", rel)
    tempos["camada4"] = time.monotonic()-inicio
    inicio = time.monotonic()
    destino = tuple(a.tamanho_saida or tamanho)
    exportar(camadas, pasta, tamanho, faixa, destino, a.inicio, a.precisao, indices,
             a.trava_fluxo, entrada, mask, a.trava_textura, a.teto_delta)
    tempos["exportacao"] = time.monotonic()-inicio
    if not a.medir_separado:
        inicio = time.monotonic()
        def original(i):
            f = normalizar(seq.quadro(i))
            if f.shape[:2] == (tamanho[1], tamanho[0]):
                return cv2.resize(f, destino, interpolation=cv2.INTER_AREA)
            return projetar(f, tamanho, faixa, destino)
        def corrigida(i):
            if i not in indices:
                return original(i)
            d = ler_cor(pasta/"soma"/f"delta_{a.inicio+i:04d}.exr")
            return np.clip(original(i)+d, 0, 1)
        def peso(i):
            w = np.ones(destino[::-1], np.float32) if regiao is None else cv2.resize(
                regiao(i), destino, interpolation=cv2.INTER_AREA)
            for exc in exclusoes:
                w[cv2.resize(exc(i), destino, interpolation=cv2.INTER_NEAREST)>0] = 0
            return w
        centros = [i for i in indices if 2 <= i < seq.quantidade - 2]
        med = medir(original, {"soma":corrigida}, peso, {"sequência": centros},
                    seq.quantidade)
        salvar_relatorios(med, pasta)
        tempos["medidas"] = time.monotonic()-inicio
    custo = recursos.fechar()
    gravar_json(pasta/"execucao.json", {"recursos":custo, "segundos_por_etapa":tempos,
                                     "tela":tamanho, "faixa":faixa, "tamanho_saida":destino,
                                     "primeiro":a.inicio, "ultimo":a.fim, "precisao":a.precisao,
                                     "entrega":[inicio_entrega, fim_entrega], "trava_fluxo":a.trava_fluxo,
                                     "trava_textura":a.trava_textura, "teto_delta":a.teto_delta,
                                     "numpy":np.__version__, "opencv":cv2.__version__,
                                     "openexr":OpenEXR.__version__})
    gravar_resumo(pasta, rel, custo)


def gravar_resumo(pasta, track_rel, custo):
    linhas = ["# Execução do Flick Canceling", "",
              f"Duração total de {custo['segundos']:.1f} segundos.", "",
              "## Confiança do registro", "",
              "| Seleção | Janelas | Média das correlações médias | Menor correlação | Falhas do ECC |",
              "|---|---:|---:|---:|---:|"]
    confianca = {}
    for chave in ("manuais", "automaticas"):
        itens = track_rel[chave]
        if not itens:
            continue
        media = float(np.mean([t["correlacao_media"] for t in itens]))
        minimo = min(t["correlacao_minima"] for t in itens)
        falhas = sum(t["falhas_ecc"] for t in itens)
        confianca[chave] = dict(janelas=len(itens), media=media, minima=minimo, falhas=falhas)
        linhas.append(f"| {chave} | {len(itens)} | {media:.4f} | {minimo:.4f} | {falhas} |")
    linhas.extend(["", "As falhas recebem correlação zero. Consulte registro4.json para os valores por quadro.",
                   "", "## Recursos", ""])
    linhas += [f"- {k}: {v}" for k,v in custo["picos_amostrados_1s"].items()]
    relatorio = {"confianca_track": confianca, "recursos": custo}
    if (pasta/"medidas.md").exists():
        linhas += ["", (pasta/"medidas.md").read_text(encoding="utf-8").removeprefix("# Medidas").strip()]
        relatorio["medidas"] = json.loads((pasta/"medidas.json").read_text(encoding="utf-8"))
    gravar_json(pasta/"relatorio.json", relatorio)
    (pasta/"relatorio.md").write_text("\n".join(linhas)+"\n", encoding="utf-8")
