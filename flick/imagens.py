"""Sequências BGR, cobertura das máscaras e entrega RGB com sinal."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import OpenEXR


def nome(padrao, quadro):
    texto = str(padrao)
    return Path(texto % quadro if "%" in texto else texto)


def normalizar(imagem):
    if np.issubdtype(imagem.dtype, np.unsignedinteger):
        return imagem.astype(np.float32) / np.iinfo(imagem.dtype).max
    if not np.issubdtype(imagem.dtype, np.floating):
        raise ValueError("Use inteiros sem sinal ou ponto flutuante para as cores.")
    return imagem.astype(np.float32, copy=False)


def ler_cor(arquivo):
    if arquivo.suffix.lower() == ".exr":
        with OpenEXR.File(str(arquivo), separate_channels=True) as exr:
            canais = exr.channels()
            if any(c not in canais for c in "BGR"):
                raise ValueError(f"Faltam os canais principais R, G e B em {arquivo}")
            return np.stack([canais[c].pixels for c in "BGR"], axis=-1)
    if arquivo.suffix.lower() not in (".png", ".tif", ".tiff"):
        raise ValueError(f"Formato de imagem desconhecido: {arquivo}")
    imagem = cv2.imread(str(arquivo), cv2.IMREAD_UNCHANGED)
    if imagem is None:
        raise ValueError(f"Falha de leitura: {arquivo}")
    if imagem.ndim != 3 or imagem.shape[2] not in (3, 4):
        raise ValueError(f"Esperada imagem RGB ou RGBA: {arquivo}")
    return np.ascontiguousarray(imagem[..., :3])


class Sequencia:
    def __init__(self, padrao, primeiro, ultimo, origem_npy=None):
        self.padrao = padrao
        self.primeiro = primeiro
        self.quantidade = ultimo - primeiro + 1
        self.matriz = None
        self.offset = 0
        if self.quantidade < 1:
            raise ValueError("Intervalo de quadros vazio.")
        if Path(padrao).suffix.lower() == ".npy":
            self.matriz = np.load(padrao, mmap_mode="r", allow_pickle=False)
            self.offset = primeiro - (origem_npy if origem_npy is not None else primeiro)
            if self.offset < 0 or self.offset + self.quantidade > len(self.matriz):
                raise ValueError("Intervalo fora do volume NPY.")
            if self.matriz.ndim != 4 or self.matriz.shape[-1] != 3:
                raise ValueError("Esperado NPY com eixos tempo, altura, largura e BGR.")
        elif self.quantidade > 1 and "%" not in str(padrao):
            raise ValueError("Uma sequência de imagens precisa do número de quadro no padrão.")

    def quadro(self, i):
        if self.matriz is not None:
            return self.matriz[self.offset+i]
        return ler_cor(nome(self.padrao, self.primeiro+i))


def crypto(arquivo, seletores, camada):
    """Soma a cobertura dos pares ID/peso. Preserva o padrão binário dos IDs float32."""
    with OpenEXR.File(str(arquivo), separate_channels=True) as exr:
        cab = exr.header()
        def texto(v):
            return v.decode("utf-8") if isinstance(v, bytes) else v
        manifestos = {}
        for k, v in cab.items():
            if k.startswith("cryptomatte/") and k.endswith("/name"):
                manifestos[texto(v)] = texto(cab.get(k.rsplit("/", 1)[0]+"/manifest", "{}"))
        if camada is None:
            if len(manifestos) != 1:
                raise ValueError("Informe a camada Cryptomatte pelo nome no cabeçalho.")
            camada = next(iter(manifestos))
        manifesto = json.loads(manifestos.get(camada, "{}"))
        ids = set()
        for seletor in seletores:
            valor = manifesto.get(seletor, seletor.removeprefix("0x"))
            if not re.fullmatch(r"[0-9a-fA-F]{8}", valor):
                raise ValueError(f"Nome ausente do manifesto ou ID inválido: {seletor}")
            bits = int(valor, 16)
            if (bits >> 23) & 255 in (0, 255):
                bits ^= 1 << 23
            ids.add(bits)
        canais = exr.channels()
        camadas = sorted({k.rsplit(".", 1)[0] for k in canais
                          if re.fullmatch(re.escape(camada)+r"[0-9]+\.[RGBA]", k, re.I)})
        if not camadas:
            raise ValueError(f"Camada Cryptomatte ausente: {camada}")
        cobertura = None
        for nivel in camadas:
            for id_, peso in (("R", "G"), ("B", "A")):
                indice = canais[nivel+"."+id_].pixels
                if indice.dtype != np.float32:
                    raise ValueError("Os canais de ID do Cryptomatte precisam ser float32.")
                pesos = canais[nivel+"."+peso].pixels
                if cobertura is None:
                    cobertura = np.zeros(indice.shape, np.float32)
                cobertura += pesos * np.isin(indice.view(np.uint32), list(ids))
        return np.clip(cobertura, 0, 1)


def leitor_mascara(padrao, primeiro, offset=0, ids=None, camada=None):
    if padrao is None:
        return None
    if Path(padrao).suffix.lower() == ".npy":
        volume = np.load(padrao, mmap_mode="r", allow_pickle=False)
        if volume.ndim not in (2, 3):
            raise ValueError("Máscara NPY precisa ter eixos [T,]H,W e valores de 0 a 1.")
        return lambda i: np.asarray(volume if volume.ndim == 2 else volume[offset+i], np.float32)
    def ler(i):
        arq = nome(padrao, primeiro+i)
        if ids:
            return crypto(arq, ids, camada)
        m = cv2.imread(str(arq), cv2.IMREAD_UNCHANGED)
        if m is None or m.ndim != 2 or m.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Esperada máscara PNG em cinza de 8 ou 16 bits: {arq}")
        return normalizar(m)
    return ler


def recorte_nativo(imagem, tamanho, faixa):
    x, y, w, h = faixa
    sx, sy = imagem.shape[1]/tamanho[0], imagem.shape[0]/tamanho[1]
    xa, ya, xb, yb = round(x*sx), round(y*sy), round((x+w)*sx), round((y+h)*sy)
    if xb <= xa or yb <= ya:
        raise ValueError("O recorte ocupa menos de um pixel na resolução da máscara.")
    return imagem[ya:yb, xa:xb]


def preparar_mascaras(regiao, exclusoes, tamanho, faixa, quantidade, pasta, params):
    if regiao is None and not exclusoes:
        return None, None
    w, h = faixa[2:]
    forma = (quantidade, h, w)
    reg_path = pasta / "regiao.npy"
    mascara = np.lib.format.open_memmap(reg_path, "w+", np.float32, forma)
    exc_path = pasta / "exclusao.npy" if exclusoes else None
    negativa = np.lib.format.open_memmap(exc_path, "w+", np.uint8, forma) if exclusoes else None
    for i in range(quantidade):
        if regiao is None:
            m = np.ones((h, w), np.float32)
        else:
            fonte = regiao(i)
            if not np.all(np.isfinite(fonte)) or fonte.min() < 0 or fonte.max() > 1:
                raise ValueError("Cobertura da região fora de 0 a 1.")
            esc = fonte.shape[1] / 1920
            m = recorte_nativo(fonte, tamanho, faixa)
            diam = max(1, round(params["dilatacao"]*esc)) | 1
            if diam > 1:
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam)))
            sigma = params["suavizacao"]*esc
            if sigma > 0:
                m = cv2.GaussianBlur(m, (0, 0), sigma)
            m = cv2.resize(np.clip(m, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
        if exclusoes:
            neg = np.zeros((h, w), bool)
            for ler in exclusoes:
                neg |= cv2.resize(recorte_nativo(ler(i), tamanho, faixa), (w, h),
                                  interpolation=cv2.INTER_NEAREST) > 0
            m[neg] = 0
            negativa[i] = neg
        mascara[i] = m
    mascara.flush()
    if negativa is not None:
        negativa.flush()
    return reg_path, exc_path


def gravar_exr(caminho, bgr, precisao="half"):
    dt = np.float16 if precisao == "half" else np.float32
    rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=dt)
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION,
                       "type": OpenEXR.scanlineimage}, {"RGB": rgb}) as arq:
        arq.write(str(caminho))
