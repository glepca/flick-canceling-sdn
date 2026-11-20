"""Trava opcional para correções em material com deslocamento ou oclusão."""

import cv2
import numpy as np

from .imagens import normalizar


def _cinza_fluxo(imagem, tamanho):
    cinza = cv2.cvtColor(cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    # Atenua variações largas de luz que poderiam puxar o registro.
    local = cinza - cv2.GaussianBlur(cinza, (0, 0), 5)
    return np.rint(np.clip(0.35 + 0.25 * cinza + 2 * local, 0, 1) * 255).astype(np.uint8)


def _peso_textura(atual, vizinho, limiar):
    """Correlação de luminância em 15 × 15 pixels, invariante a ganho e nível."""
    a = cv2.cvtColor(atual, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(vizinho, cv2.COLOR_BGR2GRAY)
    def media(canal):
        return cv2.blur(canal, (15, 15))
    ma, mb = media(a), media(b)
    va = np.maximum(media(a * a) - ma * ma, 0)
    vb = np.maximum(media(b * b) - mb * mb, 0)
    cov = media(a * b) - ma * mb
    rho = cov / np.sqrt(np.maximum(va * vb, 1e-12))
    # Sem contraste suficiente, a movimentação de uma faceta fica indeterminada.
    contraste = np.minimum(va, vb)
    return (np.clip((rho - limiar) / (1 - limiar), 0, 1) *
            np.clip((contraste - .003**2) / (.008**2 - .003**2), 0, 1))


def trava(entrada, regiao, i, delta, limite, limiar_textura=None, teto=None):
    """Cada canal só pode corrigir um extremo confirmado pelos dois vizinhos.

    As correspondências precisam voltar ao ponto de partida e permanecer dentro
    da região. O ganho cai entre metade e a totalidade do limite em px/quadro.
    """
    h, w = delta.shape[:2]
    permitida = regiao[i] > 0 if regiao is not None else np.ones((h, w), bool)
    info = {"pixels_na_regiao": int(permitida.sum()), "pixels_velozes": 0,
            "pixels_inseguros": 0, "pixels_com_ganho": 0}
    if i == 0 or i + 1 == len(entrada) or min(w, h) < 16:
        info["pixels_inseguros"] = info["pixels_na_regiao"]
        return np.zeros_like(delta), info
    escala = min(1., 720 / max(w, h))
    tamanho = (max(16, round(w * escala)), max(16, round(h * escala)))
    sx, sy = w / tamanho[0], h / tamanho[1]
    yy, xx = np.indices(tamanho[::-1], dtype=np.float32)
    y, x = np.indices((h, w), dtype=np.float32)
    atual = normalizar(entrada[i])
    atual_peq = _cinza_fluxo(atual, tamanho)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ganho = np.ones_like(delta)
    velocidade = np.zeros(tamanho[::-1], np.float32)
    confiavel = permitida.copy()
    textura = np.ones((h, w), np.float32)
    for j in (i - 1, i + 1):
        vizinho = normalizar(entrada[j])
        peq = _cinza_fluxo(vizinho, tamanho)
        ida = dis.calc(atual_peq, peq, None)
        volta = dis.calc(peq, atual_peq, None)
        retorno = ida + cv2.remap(volta, xx + ida[..., 0], yy + ida[..., 1], cv2.INTER_LINEAR)
        erro = np.hypot(retorno[..., 0], retorno[..., 1])
        velocidade = np.maximum(velocidade, np.hypot(ida[..., 0] * sx, ida[..., 1] * sy))
        confiavel &= cv2.resize(erro, (w, h)) < 0.75
        fluxo = cv2.resize(ida, (w, h))
        mapa_x = x + fluxo[..., 0] * sx
        mapa_y = y + fluxo[..., 1] * sy
        confiavel &= (mapa_x >= 0) & (mapa_x <= w - 1) & (mapa_y >= 0) & (mapa_y <= h - 1)
        if regiao is not None:
            suporte = cv2.remap((regiao[j] > 0).astype(np.float32), mapa_x, mapa_y, cv2.INTER_LINEAR)
            confiavel &= suporte > 0.999
        registrado = cv2.remap(vizinho, mapa_x, mapa_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        if limiar_textura is not None:
            np.minimum(textura, _peso_textura(atual, registrado, limiar_textura), out=textura)
        diferenca = cv2.GaussianBlur(registrado - atual, (0, 0), 1.5)
        # O sinal do quociente corta movimento monotônico; seu módulo limita
        # a soma ao menor desvio entre o centro e os dois vizinhos alinhados.
        razao = np.zeros_like(delta)
        np.divide(diferenca, delta, out=razao, where=delta != 0)
        np.minimum(ganho, np.maximum(razao, 0), out=ganho)
    velocidade = cv2.resize(velocidade, (w, h))
    rampa = np.clip(2 * (1 - velocidade / limite), 0, 1)
    ganho *= (confiavel * rampa * textura)[..., None]
    if limiar_textura is not None:
        info["pixels_vetados_textura"] = int((permitida & (textura == 0)).sum())
    if teto is not None:
        amplitude = np.max(np.abs(delta * ganho), axis=2)
        inicio = teto / 2
        comprimida = inicio * (1 + np.tanh(np.maximum(0, amplitude - inicio) / inicio))
        ganho *= np.minimum(1, comprimida / np.maximum(amplitude, 1e-12))[..., None]
    info.update(pixels_velozes=int((permitida & (velocidade >= limite)).sum()),
                pixels_inseguros=int((permitida & ~confiavel).sum()),
                pixels_com_ganho=int(np.any(ganho > 0, axis=2).sum()))
    return ganho, info
