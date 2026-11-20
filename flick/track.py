"""Camada de correção temporal no espaço de uma máscara ou de uma janela estabilizada."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import time

try:
    import resource
except ImportError:  # Indisponível no Windows.
    resource = None

import cv2
import numpy as np


# Comprimentos em pixels na largura de referência de 3840; área em pixels quadrados.
PADRAO = {
    "automatica": True, "referencia": None, "mascaras": [],
    "lado": 640, "passo": 480, "margem": 80, "rampa": 80, "peso_minimo": 0.03,
    "sigma_cinza": 2.0, "ecc_iteracoes_manual": 80, "ecc_iteracoes_auto": 60,
    "ecc_epsilon": 1e-5, "ecc_filtro": 5, "ecc_minimo": 0.6,
    "validade_minima": 0.99, "validade_erosao": 9, "validade_sigma": 4.0,
    "sigma_espacial": 2.5, "raio_mediana": 3, "sigma_temporal": 6.0,
    "extensao_temporal": 6, "limite_delta": 0.2,
    "limiar_selecao": 0.0045, "area_minima": 400, "sigma_selecao": 3.0,
    "dilatacao_manual": 41, "suavizacao_manual": 10.0,
    "dilatacao_auto": 25, "suavizacao_auto": 8.0,
    "limiar_oscilacao_relativo": 0.6, "sigma_oscilacao": 6.0, "ganho_oscilacao": 1.5,
    "trava_inicio": 0.05, "trava_fim": 0.10, "sigma_trava": 2.0,
}

_PESOS = np.array([0.0722, 0.7152, 0.2126], np.float32)


def _borrar(imagem, sigma):
    return cv2.GaussianBlur(imagem, (0, 0), sigma) if sigma > 0 else imagem


def _impar(diametro, escala):
    # Escalar o diâmetro em torno do pixel central preserva a simetria da operação.
    return 1 + 2 * round((diametro - 1) * escala / 2)


def _dimensoes(receita, escala):
    p = {**PADRAO, **receita}
    if escala <= 0 or p["trava_fim"] <= p["trava_inicio"]:
        raise ValueError("Confira a escala e os limiares da trava de movimento.")
    for chave in ("lado", "passo", "margem", "rampa"):
        p[chave] = max(int(chave != "margem"), round(p[chave] * escala))
    if p["passo"] > p["lado"]:
        raise ValueError("O passo da camada traqueada excede o lado da janela.")
    for chave in ("ecc_filtro", "validade_erosao", "dilatacao_auto", "dilatacao_manual"):
        p[chave] = max(1, _impar(p[chave], escala))
    for chave in ("sigma_cinza", "validade_sigma", "sigma_espacial", "sigma_selecao",
                  "suavizacao_manual", "suavizacao_auto", "sigma_oscilacao", "sigma_trava"):
        p[chave] *= escala
    p["area_minima"] = max(1, round(p["area_minima"] * escala**2))
    if p["raio_mediana"] < 0 or p["sigma_temporal"] < 0 or p["extensao_temporal"] <= 0:
        raise ValueError("O intervalo de suavização temporal é inválido.")
    return p


def _entrada(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _novo(path, forma, tipo):
    if Path(path).exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=tipo, shape=forma)


def _quadro_base(fonte, anteriores, indice, janela, manual):
    x, y, w, h = janela
    sl = np.s_[indice, y:y+h, x:x+w]
    imagem = fonte[sl].astype(np.float32)
    if np.issubdtype(fonte.dtype, np.unsignedinteger):
        imagem /= np.iinfo(fonte.dtype).max
    if manual and anteriores:
        delta = anteriores[0][sl].astype(np.float32)
        for a in anteriores[1:]:
            delta += a[sl]
        imagem += delta
    else:
        for a in anteriores:
            imagem += a[sl]
    return np.clip(imagem, 0, 1, out=imagem)


def _registrar(fonte, anteriores, janela, r, p, manual, low, validade):
    """Estima afins sequenciais a partir de cada lado da referência, em coordenadas locais."""
    w, h = janela[2:]
    def quadro(t):
        return _quadro_base(fonte, anteriores, t, janela, manual)

    def cinza(a):
        return _borrar(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), p["sigma_cinza"])

    ref = cinza(quadro(r))
    iteracoes = p["ecc_iteracoes_manual" if manual else "ecc_iteracoes_auto"]
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iteracoes), p["ecc_epsilon"])
    kernel = np.ones((p["validade_erosao"], p["validade_erosao"]), np.uint8)
    unidade = np.ones((h, w), np.float32)
    n = len(fonte)
    afinidades = np.empty((n, 2, 3), np.float32)
    correlacao = np.zeros(n, np.float32)
    falhou = np.zeros(n, bool)
    inverso = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    for direcao in (range(r, n), range(r, -1, -1)):
        matriz = np.eye(2, 3, dtype=np.float32)
        for t in direcao:
            a = quadro(t)
            try:
                correlacao[t], matriz = cv2.findTransformECC(
                    ref, cinza(a), matriz, cv2.MOTION_AFFINE, criterio, None, p["ecc_filtro"])
                falhou[t] = False
            except cv2.error:
                correlacao[t] = 0
                falhou[t] = True
            afinidades[t] = matriz
            estabilizado = cv2.warpAffine(a, matriz, (w, h), flags=inverso, borderMode=cv2.BORDER_REPLICATE)
            low[t] = _borrar(estabilizado, p["sigma_espacial"])
            valido = cv2.warpAffine(unidade, matriz, (w, h), flags=inverso) > p["validade_minima"]
            erodido = cv2.erode(valido.astype(np.uint8), kernel).astype(np.float32)
            v = _borrar(erodido, p["validade_sigma"])
            validade[t] = v if (manual or correlacao[t] > p["ecc_minimo"]) else 0
    return afinidades, correlacao, falhou


def _faixas(forma):
    n, h, w, _ = forma
    # A mediana materializa dados ao longo do tempo. Limite por lote e por processo.
    linhas = max(1, min(64, (16 << 20) // max(n * w * 3 * 4, 1)))
    for y in range(0, h, linhas):
        yield slice(y, min(h, y + linhas))


def _automatico(low, validade, p):
    n, h, w, _ = low.shape
    energia = np.zeros((h, w), np.float32)
    if n >= 3:
        for faixa in _faixas(low.shape):
            luma = low[:, faixa] @ _PESOS
            v = validade[:, faixa]
            alternancia = np.abs(luma[1:-1] - (luma[:-2] + luma[2:]) / 2)
            alternancia *= np.minimum(np.minimum(v[1:-1], v[:-2]), v[2:])
            energia[faixa] = np.median(alternancia, axis=0, overwrite_input=True)
    energia = _borrar(energia, p["sigma_selecao"])
    selecao = (energia > p["limiar_selecao"]).astype(np.uint8)
    _, rotulos, stats, _ = cv2.connectedComponentsWithStats(selecao)
    mantidos = stats[:, cv2.CC_STAT_AREA] >= p["area_minima"]
    mantidos[0] = False
    selecao = mantidos[rotulos].astype(np.uint8)
    osc = (energia > p["limiar_selecao"] * p["limiar_oscilacao_relativo"]).astype(np.float32)
    osc = np.clip(_borrar(osc, p["sigma_oscilacao"]) * p["ganho_oscilacao"], 0, 1)
    return selecao, osc, int(mantidos.sum())


def _trajetoria(low, p, caminho):
    """Mediana curta nas extremidades disponíveis, depois gaussiano temporal com bordas replicadas."""
    n = len(low)
    resultado = _novo(caminho, low.shape, np.float32)
    raio = int(p["raio_mediana"])
    sigma = p["sigma_temporal"]
    tamanho = int(sigma * p["extensao_temporal"]) | 1
    unidade = np.ones(1, np.float32)
    gauss = cv2.getGaussianKernel(tamanho, sigma, cv2.CV_32F) if tamanho > 1 else unidade
    for faixa in _faixas(low.shape):
        dados = np.ascontiguousarray(low[:, faixa])
        mediana = np.empty_like(dados)
        for t in range(n):
            mediana[t] = np.median(dados[max(0, t-raio):t+raio+1], axis=0)
        alvo = cv2.sepFilter2D(mediana.reshape(n, -1), -1, unidade, gauss, borderType=cv2.BORDER_REPLICATE)
        resultado[:, faixa] = np.clip(alvo.reshape(dados.shape) - dados, -p["limite_delta"], p["limite_delta"])
    return resultado


def _recursos(diagnostico, inicio):
    diagnostico["segundos"] = time.monotonic() - inicio
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        diagnostico["pico_rss_processo_bytes"] = int(rss) * (1 if sys.platform == "darwin" else 1024)
    return diagnostico


def _calcular_janela(tarefa):
    inicio = time.monotonic()
    entrada, camadas, janela, r, p, mascara, destino = tarefa
    cv2.setNumThreads(1)
    destino = Path(destino)
    fonte = _entrada(entrada)
    anteriores = [_entrada(path) for path in camadas]
    n = len(fonte)
    x, y, w, h = janela
    manual = mascara is not None
    with TemporaryDirectory(dir=destino.parent, prefix="calculo-") as tmp:
        tmp = Path(tmp)
        low = _novo(tmp / "baixas.npy", (n, h, w, 3), np.float32)
        validade = _novo(tmp / "validade.npy", (n, h, w), np.float32)
        matrizes, cc, falhou = _registrar(fonte, anteriores, janela, r, p, manual, low, validade)
        osc = None
        if manual:
            selecao = mascara
            componentes = int(cv2.connectedComponents((selecao > 0).astype(np.uint8))[0] - 1)
        else:
            selecao, osc, componentes = _automatico(low, validade, p)
        diagnostico = {
            "janela": [x, y, w, h], "fracao_selecionada": float(selecao.mean()),
            "componentes": componentes, "correlacao_media": float(cc.mean()),
            "correlacao_minima": float(cc.min()), "falhas_ecc": int(falhou.sum()),
            "correlacao_por_quadro": cc.tolist(),
        }
        if not np.any(selecao):
            return None, _recursos(diagnostico, inicio)
        sufixo = "manual" if manual else "auto"
        diametro = p["dilatacao_" + sufixo]
        elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diametro, diametro))
        m = _borrar(cv2.dilate(selecao, elemento).astype(np.float32), p["suavizacao_" + sufixo])
        alvo = _trajetoria(low, p, tmp / "diferenca.npy")
        out = _novo(destino, low.shape, np.float16)
        for t in range(n):
            d = alvo[t] * (m * validade[t])[..., None]
            if osc is not None:
                amplitude = _borrar(np.abs(d).max(axis=2), p["sigma_trava"])
                trava = np.clip((amplitude - p["trava_inicio"]) / (p["trava_fim"] - p["trava_inicio"]), 0, 1)
                d *= (1 - trava * (1 - osc))[..., None]
            out[t] = cv2.warpAffine(d, matrizes[t], (w, h), flags=cv2.INTER_LINEAR)
        out.flush()
    return str(destino), _recursos(diagnostico, inicio)


def _mascara(spec, forma, origem, margem):
    h, w = forma
    ox, oy = origem
    arquivo = Path(spec["arquivo"])
    if arquivo.suffix.lower() == ".json":
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        pontos = dados["pontos"] if isinstance(dados, dict) else dados
        pontos = np.asarray(pontos, dtype=np.float64)
        if pontos.ndim != 2 or pontos.shape[1] != 2 or len(pontos) < 3 or not np.isfinite(pontos).all():
            raise ValueError("O polígono precisa de pelo menos três pontos [x,y] finitos.")
        cheia = np.zeros((h, w), np.uint8)
        cv2.fillPoly(cheia, [(pontos - [ox, oy]).astype(np.int32)], 1)
    else:
        imagem = cv2.imread(str(arquivo), cv2.IMREAD_UNCHANGED)
        if imagem is None:
            raise ValueError(f"Leitura da máscara traqueada falhou em {arquivo}.")
        if imagem.ndim != 2:
            imagem = imagem[..., 3] if imagem.shape[2] == 4 else imagem.max(axis=2)
        if imagem.shape != (h, w):
            imagem = imagem[oy:oy+h, ox:ox+w]
        if imagem.shape != (h, w):
            raise ValueError("A dimensão da máscara traqueada precisa abranger o recorte de trabalho.")
        if imagem.dtype not in (np.uint8, np.uint16):
            raise ValueError("Use uma máscara PNG em 8 ou 16 bits.")
        cheia = imagem.astype(np.float32) / np.iinfo(imagem.dtype).max
    if not cheia.any():
        raise ValueError("A máscara traqueada está vazia dentro da faixa.")
    if "janela" in spec:
        x, y, largura, altura = map(int, spec["janela"])
        x, y = x - ox, y - oy
    else:
        x, y, largura, altura = cv2.boundingRect((cheia > 0).astype(np.uint8))
        x, y = x - margem, y - margem
        largura, altura = largura + 2*margem, altura + 2*margem
    x1, y1 = min(w, x + largura), min(h, y + altura)
    x, y = max(0, x), max(0, y)
    if x1 <= x or y1 <= y or not np.any(cheia[y:y1, x:x1]):
        raise ValueError("A janela de track e a máscara precisam se sobrepor dentro da faixa.")
    return (x, y, x1-x, y1-y), np.ascontiguousarray(cheia[y:y1, x:x1])


def _janela_auto(h, w, p):
    def eixo(n):
        ultimo = max(0, n - p["lado"])
        return sorted(set(range(0, ultimo + 1, p["passo"])) | {ultimo})
    m = p["margem"]
    for y in eixo(h):
        for x in eixo(w):
            x0, y0 = max(0, x-m), max(0, y-m)
            yield x0, y0, min(w, x+p["lado"]+m)-x0, min(h, y+p["lado"]+m)-y0


def _rampa(n, p):
    i = np.arange(n)
    return np.clip(np.minimum(i+1, n-i) / p["rampa"], p["peso_minimo"], 1).astype(np.float32)


def _ler_matte(matte, t):
    return matte[t] if matte.ndim == 3 else matte


def _referencia(q, primeiro, n):
    indice = (n-1)//2 if q is None else int(q) - primeiro
    if not 0 <= indice < n:
        raise ValueError("O quadro de referência da camada traqueada fica fora do intervalo.")
    return indice


def executar(entrada, anteriores, regiao, pasta, receita, primeiro, processos, escala=None,
             origem=(0, 0), exclusoes=None):
    """Retorna (caminho do delta BGR aditivo half, estatísticas). Entradas NPY do recorte.

    A região é o envelope suave da camada automática. A exclusão dura vale para as duas
    partes. Primeiro e referências usam números absolutos. Janela e pontos usam [x,y]
    na imagem inteira. A escala espacial é largura inteira / 3840.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "camada4.npy"
    if destino.exists():
        raise FileExistsError(destino)
    fonte = _entrada(entrada)
    n, h, w, _ = fonte.shape
    p = _dimensoes(receita, w/3840 if escala is None else escala)
    r = _referencia(p["referencia"], primeiro, n)
    relatorio = {"primeiro": primeiro, "referencia_automatica": primeiro+r, "manuais": [], "automaticas": []}
    regiao = _entrada(regiao) if regiao is not None else None
    exclusoes = _entrada(exclusoes) if exclusoes is not None else None
    for matte in (regiao, exclusoes):
        if matte is not None and matte.shape not in ((n,h,w), (h,w)):
            raise ValueError("A região e as exclusões devem ter as dimensões do recorte.")
    with TemporaryDirectory(dir=pasta, prefix="track-") as temp:
        tmp = Path(temp)
        camadas = list(anteriores)
        manual = None
        if p["mascaras"]:
            manual = _novo(tmp / "manual.npy", fonte.shape, np.float16)
            for numero, spec in enumerate(p["mascaras"]):
                janela, mascara = _mascara(spec, (h, w), origem, p["margem"])
                rm = _referencia(spec.get("quadro", p["referencia"]), primeiro, n)
                patch, info = _calcular_janela((entrada, camadas, janela, rm, p, mascara,
                                               tmp / f"mascara-{numero}.npy"))
                info.update(quadro_referencia=primeiro+rm, indice_mascara=numero)
                relatorio["manuais"].append(info)
                x, y, largura, altura = janela
                pedaco = _entrada(patch)
                for t in range(n):
                    sl = np.s_[y:y+altura, x:x+largura]
                    v = pedaco[t].astype(np.float32)
                    if exclusoes is not None:
                        v *= (_ler_matte(exclusoes, t)[sl] <= 0)[..., None]
                    manual[t][sl] = manual[t][sl].astype(np.float32) + v
                del pedaco
                Path(patch).unlink()
                print(f"Camada 4, máscara {numero+1}: correlação média {info['correlacao_media']:.3f}", flush=True)
            manual.flush()
            camadas.append(manual.filename)

        acc = None
        if p["automatica"]:
            acc = _novo(tmp / "acumulo.npy", fonte.shape, np.float16)
            pesos = np.zeros((h, w), np.float32)
            janelas = list(_janela_auto(h, w, p))
            tarefas = [(entrada, camadas, janela, r, p, None, tmp / f"janela-{i}.npy")
                       for i, janela in enumerate(janelas)]
            # Guarda por processo para mapas de 4 bytes de baixas, trajetória e validade.
            maior = max(j[2]*j[3] for j in janelas)
            bytes_ativos = n * maior * 28 + 192 * 1024**2
            limite = max(1, (12 * 1024**3) // bytes_ativos)
            workers = min(max(1, processos), len(tarefas), limite)
            relatorio["processos_automaticos"] = workers
            pool = ProcessPoolExecutor(workers) if workers > 1 else nullcontext()
            with pool as executor:
                resultados = executor.map(_calcular_janela, tarefas) if executor else map(_calcular_janela, tarefas)
                for i, (patch, info) in enumerate(resultados):
                    relatorio["automaticas"].append(info)
                    x, y, largura, altura = janelas[i]
                    sl = np.s_[y:y+altura, x:x+largura]
                    peso = _rampa(altura, p)[:, None] * _rampa(largura, p)[None, :]
                    pesos[sl] += peso
                    if patch is not None:
                        pedaco = _entrada(patch)
                        for t in range(n):
                            acc[t][sl] = acc[t][sl].astype(np.float32) + pedaco[t].astype(np.float32) * peso[..., None]
                        del pedaco
                        Path(patch).unlink()
                    print(f"Camada 4, janela {i+1}/{len(janelas)}: correlação média {info['correlacao_media']:.3f}", flush=True)
            pesos = np.maximum(pesos, 1e-4)[..., None]

        resultado = _novo(tmp / "camada4.npy", fonte.shape, np.float16)
        for t in range(n):
            delta = np.zeros((h, w, 3), np.float32) if acc is None else acc[t].astype(np.float32) / pesos
            if regiao is not None:
                delta *= _ler_matte(regiao, t)[..., None]
            if manual is not None:
                delta += manual[t]
            if exclusoes is not None:
                delta *= (_ler_matte(exclusoes, t) <= 0)[..., None]
            resultado[t] = delta
        resultado.flush()
        del resultado
        (tmp / "camada4.npy").rename(destino)
    return destino, relatorio
