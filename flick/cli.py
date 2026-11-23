"""Interface de linha de comando."""

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np

from . import blocos, track
from .decidir import decidir
from .fluxo import gravar_json, leitores, rodar
from .imagens import Sequencia, gravar_exr, ler_cor, nome, normalizar
from .medidas import medir, salvar_relatorios


def padroes():
    return {**{f"camada{i}": dict(p) for i, p in enumerate(blocos.PADROES, 1)},
            "camada4":dict(track.PADRAO), "regiao":{"dilatacao":31, "suavizacao":4.0}}


def atualizar(base, dados):
    for k, v in dados.items():
        if k not in base:
            raise ValueError(f"Parâmetro desconhecido: {k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"Esperado um grupo de parâmetros para {k}.")
            atualizar(base[k], v)
        else:
            base[k] = v


def receita(a):
    r = padroes()
    if a.receita:
        path = Path(a.receita).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        for m in config.get("camada4", {}).get("mascaras", []):
            m["arquivo"] = str(path.parent / m["arquivo"])
        atualizar(r, config)
    for par in a.param:
        chave, sep, valor = par.partition("=")
        grupo, ponto, chave = chave.partition(".")
        if not sep or not ponto:
            raise ValueError("Use --param grupo.chave=valor.")
        atualizar(r, {grupo:{chave:json.loads(valor)}})
    if a.mascara_track:
        r["camada4"]["mascaras"] = r["camada4"]["mascaras"] + [
            json.loads(item) for item in a.mascara_track]
    return r


def entrada(parser):
    parser.add_argument("--entrada", required=True, help="Padrão numerado ou volume NPY BGR.")
    parser.add_argument("--inicio", type=int, required=True)
    parser.add_argument("--fim", type=int, required=True, help="Último quadro, inclusive.")
    parser.add_argument("--origem-npy", type=int, help="Número do primeiro quadro armazenado no volume.")
    parser.add_argument("--saida", required=True, help="Pasta nova para a execução.")


def mascaras(parser):
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--regiao", help="PNG fixo, sequência PNG ou NPY de cobertura [0,1].")
    g.add_argument("--cryptomatte", help="EXR multicanal com pares de ID e cobertura.")
    parser.add_argument("--camada-crypto", help="Nome da camada, por exemplo CryptoObject.")
    parser.add_argument("--id-crypto", action="append", default=[], help="Nome ou hash hexadecimal. Repetível.")


def aplicar(a):
    seq = Sequencia(a.entrada, a.inicio, a.fim, a.origem_npy)
    if not np.isfinite(a.forca):
        raise ValueError("A força precisa ser finita.")
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=False)
    for i in range(seq.quantidade):
        q = a.inicio+i
        arquivo = nome(a.entrada, q)
        fmt = a.formato or (arquivo.suffix[1:].lower() if seq.matriz is None else "exr")
        destino = pasta/f"quadro_{q:04d}.{fmt}"
        if a.forca == 0 and seq.matriz is None and arquivo.suffix.lower() == destino.suffix:
            shutil.copyfile(arquivo, destino)
            continue
        original = seq.quadro(i)
        base = normalizar(original)
        if a.forca:
            d = ler_cor(nome(a.delta, q)).astype(np.float32)
            if d.shape != base.shape:
                raise ValueError("A prévia exige original e delta na mesma resolução.")
            base = np.clip(base + a.forca*d, 0, 1)
        if fmt == "exr":
            gravar_exr(destino, base, "float32")
        else:
            tipo = original.dtype if original.dtype in (np.uint8, np.uint16) else np.uint16
            imagem = np.rint(np.clip(base, 0, 1)*np.iinfo(tipo).max).astype(tipo)
            if seq.matriz is None and arquivo.suffix.lower() in (".png", ".tif", ".tiff"):
                fonte = cv2.imread(str(arquivo), cv2.IMREAD_UNCHANGED)
                if fonte.ndim == 3 and fonte.shape[-1] == 4:
                    imagem = np.dstack((imagem, fonte[...,3]))
            if not cv2.imwrite(str(destino), imagem):
                raise OSError(f"Erro ao gravar {destino}.")


def avaliar(a):
    cv2.setNumThreads(max(1, a.threads))
    seq, regiao = leitores(a)
    versao = Sequencia(a.corrigida or a.delta, a.inicio, a.fim, a.origem_npy)
    def original(i):
        return normalizar(seq.quadro(i))
    forma = seq.quadro(0).shape
    def corrigida(i):
        f = normalizar(versao.quadro(i))
        if f.shape != forma:
            raise ValueError("Original e versão precisam ter a mesma resolução.")
        return np.clip(original(i)+f, 0, 1) if a.delta else f
    grupos = {}
    for trecho in a.trecho:
        rotulo, intervalo = trecho.split("=", 1)
        ini, fim, passo = map(int, intervalo.split(","))
        grupos[rotulo] = range(ini-a.inicio, fim-a.inicio+1, passo)
    if not grupos:
        grupos["sequência"] = range(2, seq.quantidade-2, a.passo)
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=False)
    res = medir(original, {"corrigida":corrigida}, regiao, grupos, seq.quantidade,
                largura_fluxo=a.largura_fluxo, passo_local=a.passo_local)
    salvar_relatorios(res, pasta)


def main():
    cli = argparse.ArgumentParser(prog="flick", description="Correção localizada de cintilação.")
    sub = cli.add_subparsers(dest="comando", required=True)
    run = sub.add_parser("run", help="Calcula os deltas e as medidas de uma sequência.")
    entrada(run)
    mascaras(run)
    run.add_argument("--excluir", action="append", default=[], help="Máscara negativa, repetível.")
    run.add_argument("--faixa", type=int, nargs=4, metavar=("X","Y","L","A"))
    run.add_argument("--tela", type=int, nargs=2, metavar=("L","A"), help="Tela inteira de um NPY recortado.")
    run.add_argument("--tamanho-saida", type=int, nargs=2, metavar=("L","A"))
    run.add_argument("--entregar", type=int, nargs=2, metavar=("INICIO","FIM"),
                     help="Exporta esse intervalo e usa os demais quadros como contexto.")
    run.add_argument("--trava-fluxo", type=float, metavar="PX",
                     help="Limite opcional de velocidade em pixels da entrada por quadro.")
    run.add_argument("--trava-textura", type=float, metavar="CORRELACAO",
                     help="Com --trava-fluxo, exige textura compatível nos dois vizinhos.")
    run.add_argument("--teto-delta", type=float, metavar="VALOR",
                     help="Com --trava-fluxo, limita a maior mudança de canal antes de quantizar.")
    run.add_argument("--receita", help="JSON com parâmetros e máscaras da camada 4.")
    run.add_argument("--param", action="append", default=[], help="Substituição grupo.chave=valor JSON, repetível.")
    run.add_argument("--mascara-track", action="append", default=[], help="Objeto JSON com arquivo, quadro e janela opcional.")
    run.add_argument("--precisao", choices=("half","float32"), default="half")
    run.add_argument("--processos", type=int, default=6)
    run.add_argument("--medir-separado", action="store_true",
                     help="Adia as medidas para a execução do comparador com referência externa.")
    med = sub.add_parser("measure", help="Mede uma correção ou um delta contra o original.")
    entrada(med)
    mascaras(med)
    vers = med.add_mutually_exclusive_group(required=True)
    vers.add_argument("--corrigida")
    vers.add_argument("--delta")
    med.add_argument("--trecho", action="append", default=[], help="rótulo=início,fim,passo (quadros absolutos).")
    med.add_argument("--largura-fluxo", type=int, default=960)
    med.add_argument("--passo", type=int, default=1)
    med.add_argument("--passo-local", type=int, default=3)
    med.add_argument("--threads", type=int, default=2)
    apply = sub.add_parser("apply", help="Gera uma prévia a partir da soma com força ajustável.")
    entrada(apply)
    apply.add_argument("--delta", required=True)
    apply.add_argument("--forca", type=float, default=1.)
    apply.add_argument("--formato", choices=("png","tif","tiff","exr"))
    dec = sub.add_parser("decide", help="Revisa as camadas de um run e escolhe sozinho quantas aplicar.")
    entrada(dec)
    dec.add_argument("--execucao", required=True, help="Pasta do flick run com camada1 a camada4.")
    dec.add_argument("--ganho-min", type=float, default=2.0,
                     help="Pontos de tremor que uma camada a mais precisa tirar para entrar.")
    dec.add_argument("--detalhe-min", type=float, default=90.0, help="Detalhe mínimo mantido, em %%.")
    dec.add_argument("--fantasma-max", type=float, default=1.0,
                     help="Máximo de quadradinhos com borda suspeita, em %%.")
    dec.add_argument("--precisao", choices=("half","float32"), default="float32")
    a = cli.parse_args()
    try:
        if a.comando == "run":
            rodar(a, receita(a))
        elif a.comando == "measure":
            avaliar(a)
        elif a.comando == "decide":
            decidir(a)
        else:
            aplicar(a)
    except (OSError, ValueError, KeyError, cv2.error) as erro:
        cli.exit(1, f"Falha: {erro}\n")


if __name__ == "__main__":
    main()
