"""Medições de cintilação e detalhe com fluxo óptico na própria versão avaliada."""

from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np


LUMA_BGR = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
INDICADORES = ["pulo total", "pulo largo", "vermelho", "troca", "detalhe"]


def _imagem(leitor, indice):
    imagem = np.asarray(leitor(indice), dtype=np.float32)
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        raise ValueError("O comparador requer quadros BGR com três canais.")
    return imagem


def _verde_para_fluxo(imagem):
    return (np.clip(imagem[..., 1], 0, 1) ** 0.45 * 255).astype(np.uint8)


def _medidas_da_versao(leitor, regiao, indices, tamanho, sigma):
    largura, altura = tamanho
    grade_x, grade_y = np.meshgrid(np.arange(largura, dtype=np.float32),
                                    np.arange(altura, dtype=np.float32))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    @lru_cache(maxsize=5)
    def pequeno(indice):
        imagem = cv2.resize(_imagem(leitor, indice), tamanho, interpolation=cv2.INTER_AREA)
        verde = _verde_para_fluxo(imagem)
        # O DIS exige dimensões mínimas. O preenchimento só afeta imagens minúsculas.
        if min(tamanho) < 16:
            verde = cv2.copyMakeBorder(verde, 0, max(0, 16 - altura),
                                       0, max(0, 16 - largura), cv2.BORDER_REPLICATE)
        return imagem, verde

    valores = {}
    for i in indices:
        atual, verde = pequeno(i)
        if regiao is None:
            peso = np.ones((altura, largura), dtype=np.float32)
        else:
            mascara = np.asarray(regiao(i), dtype=np.float32)
            if mascara.ndim != 2:
                raise ValueError("A região do comparador deve ter um canal.")
            peso = cv2.resize(mascara, tamanho, interpolation=cv2.INTER_AREA)
        area = peso.sum()
        if area <= 0:
            continue
        vizinhos = []
        for j in (i - 2, i - 1, i + 1, i + 2):
            vizinho, verde_vizinho = pequeno(j)
            fluxo = dis.calc(verde, verde_vizinho, None)[:altura, :largura]
            vizinhos.append(cv2.remap(vizinho, grade_x + fluxo[..., 0],
                                      grade_y + fluxo[..., 1], cv2.INTER_LINEAR))
        amostras = np.stack(vizinhos + [atual])
        residuo = atual - np.median(amostras, axis=0, overwrite_input=True)
        largo = cv2.GaussianBlur(residuo, (0, 0), sigma)
        mapas = (residuo @ LUMA_BGR, largo @ LUMA_BGR, largo[..., 2],
                 (atual - vizinhos[2]) @ LUMA_BGR,
                 cv2.Laplacian(atual @ LUMA_BGR, cv2.CV_32F))
        valores[i] = [1000 * float((np.abs(mapa) * peso).sum() / area) for mapa in mapas]
    return valores


def _detalhe_medio_local(leitor, i, janela):
    luma = _imagem(leitor, i) @ LUMA_BGR
    return cv2.blur(np.abs(cv2.Laplacian(luma, cv2.CV_32F)), (janela, janela))


def medir(original, corrigidas, regiao, trechos, quantidade, *, largura_fluxo=960, passo_local=3):
    """Recebe leitores de quadros BGR float32, região e índices relativos ao primeiro.

    Os trechos só contêm centros com dois vizinhos em cada direção. As cinco
    medidas usam a região. O máximo local cobre o quadro inteiro, inclusive bordas.
    """
    if quantidade < 1 or passo_local < 1 or largura_fluxo < 1:
        raise ValueError("Quantidade, passo local e largura do fluxo devem ser positivos.")
    if "original" in corrigidas:
        raise ValueError("O nome 'original' é reservado à referência.")
    trechos = {rotulo: list(indices) for rotulo, indices in trechos.items()}
    indices = sorted({i for lista in trechos.values() for i in lista})
    if any(i < 2 or i >= quantidade - 2 for i in indices):
        raise ValueError("Cada centro precisa ter os vizinhos t-2, t-1, t+1 e t+2.")
    altura, largura = _imagem(original, 0).shape[:2]
    largura_reduzida = min(largura, largura_fluxo)
    tamanho = (largura_reduzida, max(1, round(altura * largura_reduzida / largura)))
    sigma = 4 * largura_reduzida / 960
    janela = max(1, round(64 * largura / 1920))
    resultado = {
        "indicadores": INDICADORES.copy(),
        "amostragem": {"tamanho_fluxo": list(tamanho), "sigma_largo": sigma,
                       "tamanho_local": [largura, altura], "janela_local": janela,
                       "passo_local": passo_local, "piso_local": 0.002},
        "trechos": {}, "detalhe_local": {},
    }
    for trecho, lista in trechos.items():
        resultado["trechos"][trecho] = {"indices": lista, "indices_com_regiao": [],
                                        "medias_x1000": {}, "percentuais": {}}

    for nome, leitor in {"original": original, **corrigidas}.items():
        valores = _medidas_da_versao(leitor, regiao, indices, tamanho, sigma)
        for rotulo, lista in trechos.items():
            dados = resultado["trechos"][rotulo]
            validos = [i for i in lista if i in valores]
            dados["indices_com_regiao"] = validos
            medias = np.mean([valores[i] for i in validos], axis=0).tolist() if validos else [None] * 5
            dados["medias_x1000"][nome] = medias
            if nome != "original":
                base = dados["medias_x1000"]["original"]
                dados["percentuais"][nome] = [
                    None if ref is None or ref == 0 else
                    round((valor / ref - 1) * (100 if k == 4 else -100), 1)
                    for k, (ref, valor) in enumerate(zip(base, medias))
                ]

    for i in range(0, quantidade, passo_local):
        base = _detalhe_medio_local(original, i, janela) + 0.002
        for nome, leitor in corrigidas.items():
            fator = float((_detalhe_medio_local(leitor, i, janela) / base).max())
            anterior = resultado["detalhe_local"].get(nome)
            if anterior is None or fator > anterior["fator"]:
                resultado["detalhe_local"][nome] = {"fator": fator, "indice": i}
    return resultado


def markdown_medidas(resultado):
    linhas = ["# Medidas", "",
              "Os quatro primeiros indicadores mostram redução em %. O detalhe mostra a variação com sinal. "
              "Cada percentual usa a média das medidas dos quadros válidos do trecho. "
              "O símbolo n/d indica região vazia ou referência zero.", "",
              "| Trecho | Versão | Centros válidos | Pulo total | Pulo largo | Vermelho | Troca | Detalhe |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for trecho, dados in resultado["trechos"].items():
        for nome, valores in dados["percentuais"].items():
            numeros = ["n/d" if v is None else f"{v:.1f}".replace(".", ",") for v in valores]
            linhas.append("| " + " | ".join([trecho, nome, str(len(dados["indices_com_regiao"])), *numeros]) + " |")
    linhas.extend(["", "## Maior aumento local de detalhe", "",
                   "Maior razão entre médias locais do valor absoluto do Laplaciano, sobre o quadro inteiro. "
                   "O denominador é a média no original mais 0,002. O índice começa em zero.", ""])
    for nome, dados in resultado["detalhe_local"].items():
        fator = f"{dados['fator']:.3f}".replace(".", ",")
        linhas.append(f"- {nome} atingiu {fator} vezes no índice {dados['indice']}.")
    return "\n".join(linhas) + "\n"


def salvar_relatorios(resultado, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, conteudo in (("medidas.json", json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("medidas.md", markdown_medidas(resultado))):
        with (pasta / nome).open("x", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
