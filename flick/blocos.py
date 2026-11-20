"""Cancelamento por blocos temporais sobrepostos com memória de aparência."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import multiprocessing
import os
import sys

import cv2
import numpy as np


# Distâncias e áreas em pixels de uma imagem com 3840 pixels de largura.
_BASE = dict(
    largura_referencia=3840, quadrado=480, passo=320, margem=60,
    borda_mescla=80, piso_mescla=0.03, deslocamento=0, bloco=24,
    passo_temporal=16, deslocamento_temporal=0, selecao="luma",
    pesos_bgr=[0.0722, 0.7152, 0.2126], percentis=[40, 55, 70, 80, 88, 94],
    indice_semente=2, trocas_minimas=3, janela_troca=5, extensao=21,
    fechamento=9, area_minima=60, selecao_minima=80,
    dilatacao=9, sigma_mascara=3., sigma_luma=1., sigma_cor=2.5,
    sigma_registro=1.5, filtro_ecc=5, iteracoes_ecc=60, tolerancia_ecc=1e-5,
    trava=True, histerese=0.02, cruzamentos=2, sigma_oscilacao=6.,
    ganho_oscilacao=1.5, cobertura_minima=0.99, sigma_cobertura=3.,
    trava_inicio=0.06, trava_fim=0.12, limite=0.25,
    porta=True, porta_inicio=0.004, porta_fim=0.012, sigma_porta=2.,
    percentil_alternancia=50, limiar_alternancia=0.004, sigma_alternancia=2.,
)
PADROES = [
    dict(_BASE),
    dict(_BASE, deslocamento=160, deslocamento_temporal=8, sigma_cor=5.,
         sigma_luma=2., extensao=31, area_minima=150),
    dict(_BASE, selecao="alternancia", deslocamento=80, deslocamento_temporal=4,
         sigma_cor=3., sigma_luma=1.5, area_minima=100, porta_inicio=0.002,
         porta_fim=0.005),
]


def _escalar(parametros, largura):
    p = dict(parametros)
    escala = largura / p["largura_referencia"]
    for chave in ("quadrado", "passo", "margem", "deslocamento", "borda_mescla"):
        p[chave] = max(0, round(p[chave] * escala))
    for chave in ("quadrado", "passo", "borda_mescla"):
        p[chave] = max(1, p[chave])
    for chave in ("janela_troca", "extensao", "fechamento", "dilatacao", "filtro_ecc"):
        # Núcleos centrados ímpares. Arredonda o raio geométrico.
        p[chave] = 2 * max(0, round((p[chave] - 1) * escala / 2)) + 1
    for chave in ("area_minima", "selecao_minima"):
        p[chave] = max(1, round(p[chave] * escala ** 2))
    for chave in list(p):
        if chave.startswith("sigma_"):
            p[chave] *= escala
            if p[chave] < 0:
                raise ValueError(f"Desfoque negativo em {chave}")
    if p["bloco"] < 1 or not 1 <= p["passo_temporal"] <= p["bloco"]:
        raise ValueError("O passo temporal precisa estar entre 1 e o tamanho do bloco")
    if p["passo"] > p["quadrado"] or not 0 <= p["deslocamento"] <= p["quadrado"]:
        raise ValueError("Passo e deslocamento espaciais devem manter cobertura da grade")
    if not 0 <= p["deslocamento_temporal"] <= p["bloco"]:
        raise ValueError("Deslocamento temporal fora do bloco")
    if p["porta_fim"] <= p["porta_inicio"] or p["trava_fim"] <= p["trava_inicio"]:
        raise ValueError("Cada limiar final deve superar o limiar inicial")
    if p["selecao"] not in ("luma", "alternancia"):
        raise ValueError("A seleção precisa ser luma ou alternancia")
    p["pesos_bgr"] = np.asarray(p["pesos_bgr"], dtype=np.float32)
    return p


def _inicios(tamanho, janela, passo, deslocamento):
    fim = max(0, tamanho - janela)
    return sorted({0, fim, *range(deslocamento, fim + 1, passo)})


def _blur(imagem, sigma):
    return cv2.GaussianBlur(imagem, (0, 0), sigma) if sigma > 0 else imagem


def _nucleo(n):
    return np.ones((n, n), np.uint8)


def _registrar(quadros, p):
    cinzas = [_blur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), p["sigma_registro"]) for f in quadros]
    centro = len(quadros) // 2
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                p["iteracoes_ecc"], p["tolerancia_ecc"])
    matrizes = [None] * len(quadros)
    correlacoes = [None] * len(quadros)
    for ordem in (range(centro, len(quadros)), range(centro, -1, -1)):
        matriz = np.eye(2, 3, dtype=np.float32)
        for t in ordem:
            try:
                cc, matriz = cv2.findTransformECC(
                    cinzas[centro], cinzas[t], matriz, cv2.MOTION_AFFINE,
                    criterio, None, p["filtro_ecc"])
                correlacoes[t] = float(cc)
            except cv2.error:
                correlacoes[t] = None
            matrizes[t] = matriz.copy()
    return matrizes, correlacoes


def _estabilizar(imagem, matriz):
    return cv2.warpAffine(imagem, matriz, imagem.shape[1::-1],
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _selecionar(luma, p, nucleo):
    total, h, w = luma.shape
    if p["selecao"] == "alternancia":
        if total < 3:
            return np.zeros((h, w), np.uint8)
        diferenca = luma[1:-1] - (luma[:-2] + luma[2:]) / 2
        variacao = np.percentile(np.abs(diferenca), p["percentil_alternancia"], axis=0)
        selecao = (_blur(variacao.astype(np.float32), p["sigma_alternancia"]) >
                   p["limiar_alternancia"]).astype(np.uint8)
    else:
        semente = np.zeros((h, w), np.uint8)
        aberto = np.zeros_like(semente)
        for i, limiar in enumerate(np.percentile(luma[total // 2], p["percentis"])):
            binario = (luma > limiar).astype(np.uint8)
            ampliado = [cv2.dilate(b, nucleo) for b in binario]
            contagem = np.zeros((h, w), np.int32)
            for t in range(1, total):
                contagem += ((binario[t] & (1 - ampliado[t - 1])) |
                             (binario[t - 1] & (1 - ampliado[t])))
            destino = semente if i >= p["indice_semente"] else aberto
            destino |= (contagem >= p["trocas_minimas"]).astype(np.uint8)
        aberto = cv2.morphologyEx(aberto | semente, cv2.MORPH_CLOSE, nucleo)
        elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["extensao"],) * 2)
        selecao = (aberto & cv2.dilate(semente, elipse)) | semente
    selecao = cv2.morphologyEx(selecao, cv2.MORPH_CLOSE, _nucleo(p["fechamento"]))
    _, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(selecao)
    manter = (estatisticas[:, cv2.CC_STAT_AREA] >= p["area_minima"]).astype(np.uint8)
    manter[0] = 0
    return manter[rotulos]


def _oscilacao(cor, p):
    luma = cor @ p["pesos_bgr"]
    centro = np.median(luma, axis=0)
    estado = np.zeros(centro.shape, np.int8)
    cruzamentos = np.zeros(centro.shape, np.int32)
    for imagem in luma:
        dif = imagem - centro
        novo = np.where(dif > p["histerese"], 1,
                        np.where(dif < -p["histerese"], -1, estado)).astype(np.int8)
        cruzamentos += (novo != estado) & (estado != 0)
        estado = novo
    oscila = (cruzamentos >= p["cruzamentos"]).astype(np.float32)
    return np.clip(_blur(oscila, p["sigma_oscilacao"]) * p["ganho_oscilacao"], 0, 1)


def _cobertura(uns, matriz, p, nucleo):
    projetado = cv2.warpAffine(uns, matriz, uns.shape[1::-1],
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valido = cv2.erode((projetado > p["cobertura_minima"]).astype(np.uint8), nucleo)
    return _blur(valido.astype(np.float32), p["sigma_cobertura"])


def _bloco(raw, anterior, p):
    matrizes, correlacoes = _registrar(raw, p)
    estavel = np.stack([_estabilizar(f, m) for f, m in zip(raw, matrizes)])
    luma = np.stack([_blur(f @ p["pesos_bgr"], p["sigma_luma"]) for f in estavel])
    nucleo = _nucleo(p["janela_troca"])
    selecao = _selecionar(luma, p, nucleo)
    area = int(selecao.sum())
    if area < p["selecao_minima"]:
        return None, -1, area, correlacoes
    mascara = _blur(cv2.dilate(selecao, _nucleo(p["dilatacao"])).astype(np.float32),
                    p["sigma_mascara"])[..., None]
    cor = np.stack([_blur(f, p["sigma_cor"]) for f in estavel])
    del estavel, luma
    if anterior is not None:
        comum = [_blur(_estabilizar(np.clip(raw[i] + d, 0, 1), matrizes[i]),
                       p["sigma_cor"]) for i, d in enumerate(anterior)]
        alvo = np.median(np.stack(comum), axis=0)
    else:
        alvo = np.median(cor, axis=0)
    dentro = selecao.astype(bool)
    ref = int(np.argmin([np.abs(f - alvo)[dentro].mean() for f in cor]))
    if p["trava"]:
        oscila = _oscilacao(cor, p)
        uns = np.ones(selecao.shape, np.float32)
        valida_ref = _cobertura(uns, matrizes[ref], p, nucleo)
    contra = np.zeros_like(cor)
    for t, f in enumerate(cor):
        if t == ref:
            continue
        d = np.clip(cor[ref] - f, -p["limite"], p["limite"]) * mascara
        if p["porta"]:
            amplitude = _blur(np.abs(d).max(2), p["sigma_porta"])
            d *= np.clip((amplitude - p["porta_inicio"]) /
                         (p["porta_fim"] - p["porta_inicio"]), 0, 1)[..., None]
        if p["trava"]:
            amplitude = _blur(np.abs(d).max(2), p["sigma_porta"])
            forte = np.clip((amplitude - p["trava_inicio"]) /
                            (p["trava_fim"] - p["trava_inicio"]), 0, 1)
            valida = np.minimum(valida_ref, _cobertura(uns, matrizes[t], p, nucleo))
            d *= ((1 - forte * (1 - oscila)) * valida)[..., None]
        contra[t] = cv2.warpAffine(d, matrizes[t], selecao.shape[::-1],
                                   flags=cv2.INTER_LINEAR)
    return contra, ref, area, correlacoes


def _recorte(args):
    entrada, anteriores, fatia, arquivo, primeiro, p = args
    cv2.setNumThreads(1)
    y0, y1, x0, x1 = fatia
    dados = np.load(entrada, mmap_mode="r")[:, y0:y1, x0:x1]
    base = [np.load(c, mmap_mode="r")[:, y0:y1, x0:x1] for c in anteriores]
    total, h, w, _ = dados.shape
    n = min(p["bloco"], total)
    inicios = _inicios(total, n, p["passo_temporal"], p["deslocamento_temporal"])
    pesos = np.minimum(np.arange(n) + 1, n - np.arange(n)).astype(np.float32)
    pesos /= pesos.max()
    imagem = np.lib.format.open_memmap(arquivo, mode="w+", dtype=np.float16, shape=dados.shape)
    soma = np.zeros((n, h, w, 3), np.float32)
    peso = np.zeros(n, np.float32)
    memoria = None
    referencias, areas, ecc = [], [], []
    com_memoria = falhas = 0
    for i, inicio in enumerate(inicios):
        sl = slice(inicio, inicio + n)
        raw = dados[sl].astype(np.float32)
        if np.issubdtype(dados.dtype, np.integer):
            raw /= np.iinfo(dados.dtype).max
        if base:
            anterior = base[0][sl].astype(np.float32)
            for camada in base[1:]:
                anterior += camada[sl].astype(np.float32)
            if len(base) > 1:
                anterior = anterior.astype(np.float16).astype(np.float32)
            np.clip(raw + anterior, 0, 1, out=raw)
            del anterior
        contra, ref, area, cc = _bloco(raw, memoria, p)
        areas.append(area)
        referencias.append(primeiro + inicio + ref if ref >= 0 else None)
        falhas += cc.count(None)
        ecc.extend(v for v in cc if v is not None)
        if memoria is not None and ref >= 0:
            com_memoria += 1
        salto = inicios[i + 1] - inicio if i + 1 < len(inicios) else n
        memoria = None
        if contra is not None:
            if salto < n:
                memoria = contra[salto:].copy()
            soma += contra * pesos[:, None, None, None]
            del contra
        peso += pesos
        for j in range(salto):
            imagem[inicio + j] = soma[j] / peso[j]
        soma[:n-salto] = soma[salto:]
        soma[n-salto:] = 0
        peso[:n-salto] = peso[salto:]
        peso[n-salto:] = 0
    imagem.flush()
    pico_bytes = None
    if os.name == "posix":
        import resource
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pico_bytes = int(pico if sys.platform == "darwin" else pico * 1024)
    return dict(pico_rss_processo_bytes=pico_bytes, referencias=referencias, pixels_selecionados=areas,
                blocos_com_memoria=com_memoria, falhas_ecc=falhas,
                ecc_medio=float(np.mean(ecc)) if ecc else None)


def _borda(n, p):
    i = np.arange(n)
    return np.clip(np.minimum(i + 1, n - i) / p["borda_mescla"],
                   p["piso_mescla"], 1).astype(np.float32)


def _fundir(destino, pesos, fatia, arquivo, p):
    y0, y1, x0, x1 = fatia
    parte = np.load(arquivo, mmap_mode="r")
    janela = _borda(y1-y0, p)[:, None] * _borda(x1-x0, p)[None, :]
    pesos[y0:y1, x0:x1] += janela
    for i in range(len(destino)):
        atual = destino[i, y0:y1, x0:x1]
        atual[:] = atual.astype(np.float32) + parte[i].astype(np.float32) * janela[..., None]
    del parte
    Path(arquivo).unlink()


def executar(entrada, mascara, anteriores, saida, parametros, processos=6,
             largura_original=None, primeiro=0):
    """Gera um delta aditivo half no recorte. As contas intermediárias são float32.

    A máscara opcional contém a cobertura final do recorte em [0, 1].
    Os deltas anteriores são lidos individualmente na ordem de produção.
    O arquivo de saída só aparece após concluir a camada.
    """
    saida = Path(saida)
    if saida.exists():
        raise FileExistsError(saida)
    dados = np.load(entrada, mmap_mode="r")
    if dados.ndim != 4 or dados.shape[-1] != 3 or min(dados.shape) < 1:
        raise ValueError("Esperado um arranjo não vazio N,H,W,3 em BGR")
    total, altura, largura, _ = dados.shape
    if any(np.load(a, mmap_mode="r").shape != dados.shape for a in anteriores):
        raise ValueError("As camadas anteriores devem ter o mesmo formato da entrada")
    m = np.load(mascara, mmap_mode="r") if mascara is not None else None
    if m is not None and m.shape != dados.shape[:3]:
        raise ValueError("A cobertura deve ter o mesmo número de quadros e tamanho do recorte")
    p = _escalar(parametros, largura_original or largura)
    ys = _inicios(altura, p["quadrado"], p["passo"], p["deslocamento"])
    xs = _inicios(largura, p["quadrado"], p["passo"], p["deslocamento"])
    # Limite conservador por bloco para os arranjos, máscaras e cópias temporárias.
    lado = p["quadrado"] + 2 * p["margem"]
    pixels = min(altura, lado) * min(largura, lado)
    reserva = pixels * min(total, p["bloco"]) * 160 + 128 * 1024**2
    try:
        fisica = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        fisica = 8 * 1024**3
    if processos < 1:
        raise ValueError("Escolha ao menos um processo")
    nproc = min(processos, len(xs) * len(ys), max(1, int(fisica * .45 / reserva)))
    rel = {}
    with TemporaryDirectory(dir=saida.parent, prefix=".blocos-") as temp:
        pasta = Path(temp)
        acum = np.lib.format.open_memmap(pasta / "camada.npy", mode="w+",
                                        dtype=np.float16, shape=dados.shape)
        pesos = np.zeros((altura, largura), np.float32)
        ocupado = None
        if m is not None:
            ocupado = np.zeros((altura, largura), bool)
            for quadro in m:
                ocupado |= quadro != 0
        tarefas = []
        for y in ys:
            for x in xs:
                fatia = (max(0, y - p["margem"]), min(altura, y + p["quadrado"] + p["margem"]),
                         max(0, x - p["margem"]), min(largura, x + p["quadrado"] + p["margem"]))
                if ocupado is not None and not ocupado[fatia[0]:fatia[1], fatia[2]:fatia[3]].any():
                    continue
                caminho = pasta / f"recorte-{y}-{x}.npy"
                tarefas.append((entrada, anteriores, fatia, caminho, primeiro, p))
        # Janela de envio limitada. A fusão em ordem fixa é reprodutível entre processos.
        contexto = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=contexto) as pool:
            pendentes = {}
            for i in range(min(nproc, len(tarefas))):
                pendentes[i] = pool.submit(_recorte, tarefas[i])
            for i, tarefa in enumerate(tarefas):
                resumo = pendentes.pop(i).result()
                j = i + nproc
                if j < len(tarefas):
                    pendentes[j] = pool.submit(_recorte, tarefas[j])
                fatia, caminho = tarefa[2:4]
                _fundir(acum, pesos, fatia, caminho, p)
                rel[caminho.stem] = resumo
                print(f"Quadradinho {i+1}/{len(tarefas)}", flush=True)
        np.maximum(pesos, 1e-4, out=pesos)
        for i in range(total):
            d = acum[i].astype(np.float32) / pesos[..., None]
            if m is not None:
                d *= m[i].astype(np.float32)[..., None]
            acum[i] = d
        acum.flush()
        del acum
        os.link(pasta / "camada.npy", saida)
    return dict(processos=nproc, quadradinhos=rel)
