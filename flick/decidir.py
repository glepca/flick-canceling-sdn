"""Modo automático: a ferramenta revisa o próprio resultado e decide quantas camadas aplicar.

Mede o original e cada acumulado (1, 1+2, 1+2+3, 1+2+3+4) só nos pixels em que alguma camada
atuou. Um acumulado é aceitável quando mantém o detalhe mínimo e não mostra bordas suspeitas
(quadradinhos cujo detalhe cresce demais, sinal de contorno dobrado). Entre os aceitáveis,
fica o menor acumulado que chega perto do melhor: camada que quase não ajuda não entra.
As camadas não são monotônicas, por isso todas são medidas antes da decisão.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .fluxo import gravar_json
from .imagens import Sequencia, gravar_exr, ler_cor, nome, normalizar

LADO = 64
RAZAO_FANTASMA = 1.15


def _luma(bgr):
    return (0.0722*bgr[..., 0] + 0.7152*bgr[..., 1] + 0.2126*bgr[..., 2])*255


def _ladrilhos(a):
    h, w = (a.shape[0]//LADO)*LADO, (a.shape[1]//LADO)*LADO
    if h == 0 or w == 0:
        return a.mean(keepdims=True).reshape(1, 1)
    return a[:h, :w].reshape(h//LADO, LADO, w//LADO, LADO).mean(axis=(1, 3))


def _delta(pasta, camadas, q):
    return sum(ler_cor(Path(nome(str(pasta/f"camada{k}"/"delta_%04d.exr"), q))).astype(np.float32)
               for k in camadas)


def _medir(lumas, mascara, mascara_t, lap_ref=None):
    janela, d2, det, fant, laps = [], [], [], [], []
    for i, L in enumerate(lumas):
        janela = (janela + [L])[-3:]
        if len(janela) == 3:
            d2.append(float(np.abs(janela[2] - 2*janela[1] + janela[0])[mascara].mean()))
        lap = np.abs(cv2.Laplacian(L, cv2.CV_32F))
        det.append(float(lap[mascara].mean()))
        t = _ladrilhos(lap)
        laps.append(t)
        if lap_ref is not None and mascara_t.any():
            fant.append(float((t > RAZAO_FANTASMA*lap_ref[i] + 0.5)[mascara_t].mean()*100))
    return {"tremor": float(np.mean(d2)), "detalhe": float(np.mean(det)),
            "fantasma_%": float(np.mean(fant)) if fant else 0.0}, laps


def decidir(a):
    seq = Sequencia(a.entrada, a.inicio, a.fim, a.origem_npy)
    if seq.quantidade < 3:
        raise ValueError("O modo automático precisa de pelo menos 3 quadros.")
    execucao = Path(a.execucao)
    camadas = [k for k in (1, 2, 3, 4) if (execucao/f"camada{k}").is_dir()]
    if not camadas:
        raise ValueError(f"Nenhuma pasta camada1 a camada4 em {execucao}.")
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=False)
    Q = [a.inicio + i for i in range(seq.quantidade)]
    original = lambda i: normalizar(seq.quadro(i))

    mascara = None
    for i, q in enumerate(Q):
        for k in camadas:
            m = np.abs(_delta(execucao, [k], q)).max(axis=-1) > 0
            mascara = m if mascara is None else mascara | m
    if not mascara.any():
        raise ValueError("Nenhuma camada atuou nesse intervalo.")
    mascara_t = _ladrilhos(mascara.astype(np.float32)) > 0.5

    base, lap_ref = _medir((_luma(original(i)) for i in range(len(Q))), mascara, mascara_t)
    passos = []
    for n in range(1, len(camadas) + 1):
        usadas = camadas[:n]
        m, _ = _medir((_luma(np.clip(original(i) + _delta(execucao, usadas, q), 0, 1))
                       for i, q in enumerate(Q)), mascara, mascara_t, lap_ref)
        red = 100*(1 - m["tremor"]/base["tremor"]) if base["tremor"] > 0 else 0.0
        det = 100*m["detalhe"]/base["detalhe"] if base["detalhe"] > 0 else 100.0
        veto = []
        if det < a.detalhe_min:
            veto.append(f"detalhe caiu para {det:.1f}%")
        if m["fantasma_%"] > a.fantasma_max:
            veto.append(f"{m['fantasma_%']:.2f}% dos quadradinhos com borda suspeita")
        passos.append({"camadas": usadas, "reducao_tremor_%": round(red, 1),
                       "detalhe_mantido_%": round(det, 1), "fantasma_%": round(m["fantasma_%"], 2),
                       "aceitavel": not veto, "veto": veto})

    ok = [p for p in passos if p["aceitavel"] and p["reducao_tremor_%"] > 0]
    if ok:
        melhor = max(p["reducao_tremor_%"] for p in ok)
        escolha = min((p for p in ok if p["reducao_tremor_%"] >= melhor - a.ganho_min),
                      key=lambda p: len(p["camadas"]))
        usadas = escolha["camadas"]
        conclusao = (f"Aplica as camadas {', '.join(map(str, usadas))}: tremor "
                     f"{escolha['reducao_tremor_%']}% menor, {escolha['detalhe_mantido_%']}% do detalhe.")
    else:
        usadas, conclusao = [], "Nenhuma combinação compensou sem perder detalhe; o delta escolhido é zero."

    (pasta/"escolha").mkdir()
    for i, q in enumerate(Q):
        d = _delta(execucao, usadas, q) if usadas else np.zeros_like(original(i))
        gravar_exr(pasta/"escolha"/f"delta_{q:04d}.exr", d, a.precisao)
    relatorio = {"criterios": {"ganho_min_pontos": a.ganho_min, "detalhe_min_%": a.detalhe_min,
                               "fantasma_max_%": a.fantasma_max},
                 "area_atuada_%": round(100*float(mascara.mean()), 1), "original": base,
                 "passos": passos, "camadas_escolhidas": usadas, "conclusao": conclusao}
    gravar_json(pasta/"decisao.json", relatorio)
    print(conclusao)
    return relatorio
