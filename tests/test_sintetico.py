"""Prova sintética pela CLI: a cena limpa fornece a verdade por pixel e por quadro."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import OpenEXR


def gerar_sequencia(pasta, quantidade=32):
    """Sobrepõe duas lâminas translúcidas ao fundo e injeta pulsos de durações conhecidas."""
    altura, largura = 80, 128
    y, x = np.mgrid[:altura, :largura].astype(np.float32)
    textura = 0.025 * np.cos(x * 0.44 + y * 0.17) + 0.015 * np.sin(x * 0.12 - y * 0.51)
    fundo = np.stack([0.32 + textura, 0.34 + textura, 0.30 + textura], axis=-1)
    limpos, sujos, mascaras = [], [], []
    for t in range(quantidade):
        imagem = fundo.copy()
        mancha = np.zeros_like(imagem)
        ativa = np.zeros((altura, largura), bool)
        objetos = [
            (42 + t * .11, 36, (0.61, 0.51, 0.37), t % 4 == 1, (0.14, 0.11, 0.09)),
            (86 - t * .08, 45, (0.42, 0.56, 0.68), t % 5 in (3, 4), (-0.09, -0.10, -0.12)),
        ]
        for cx, cy, cor, ligada, amplitude in objetos:
            u, v = x - cx, y - cy
            raio = np.sqrt((u / 29)**2 + (v / 23)**2)
            cobertura = (0.72 * np.clip((1.02 - raio) * 10, 0, 1))[..., None]
            # Textura local e um reflexo de trajetória contínua pertencem à cena limpa.
            reflexo = 0.055 * np.exp(-((u - .22*t + 8)**2 / 26 + (v + 4)**2 / 180))
            material = np.asarray(cor, np.float32) + (textura + reflexo)[..., None]
            imagem = imagem * (1 - cobertura) + material * cobertura
            if ligada:
                perfil = np.exp(-0.5 * ((u / 7)**2 + (v / 6)**2))
                ativa |= perfil > 0.20
                mancha += np.asarray(amplitude, np.float32) * perfil[..., None]
        limpa = np.rint(imagem * 65535).astype(np.uint16)
        suja = np.rint(np.clip(imagem + mancha, 0, 1) * 65535).astype(np.uint16)
        if not cv2.imwrite(str(pasta / f"entrada_{t+1:04d}.png"), suja):
            raise OSError("Falha ao escrever quadro sintético")
        limpos.append(limpa.astype(np.float32) / 65535)
        sujos.append(suja.astype(np.float32) / 65535)
        mascaras.append(ativa)
    return np.stack(limpos), np.stack(sujos), np.stack(mascaras)


def ler_exr(arquivo):
    with OpenEXR.File(str(arquivo), separate_channels=True) as exr:
        return np.stack([exr.channels()[k].pixels for k in "BGR"], axis=-1).astype(np.float32)


class TesteSintetico(unittest.TestCase):
    def cli(self, *args):
        destino = Path(sys.executable).with_name("flick.exe" if sys.platform == "win32" else "flick")
        r = subprocess.run([str(destino), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_sequencia_contra_verdade_conhecida(self):
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario)
            limpa, suja, manchas = gerar_sequencia(pasta)
            origem = pasta / "entrada_%04d.png"
            saida = pasta / "correcao"
            quadros = ["--entrada", origem, "--inicio", 1, "--fim", len(limpa)]
            self.cli("run", *quadros, "--saida", saida, "--processos", 2)
            deltas = []
            for q in range(1, len(limpa) + 1):
                nome = f"delta_{q:04d}.exr"
                total = ler_exr(saida / "soma" / nome)
                partes = [ler_exr(saida / f"camada{k}" / nome) for k in range(1, 5)]
                self.assertTrue(np.array_equal(total, sum(partes).astype(np.float16)))
                acao = cv2.imread(str(saida / "acao" / f"acao_{q:04d}.png"), cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.array_equal(acao > 0, np.any(np.stack(partes) != 0, axis=(0, 3))))
                deltas.append(total)
            corrigida = np.clip(suja + np.stack(deltas), 0, 1)
            erro_antes = np.abs(suja - limpa)[manchas].mean()
            erro_depois = np.abs(corrigida - limpa)[manchas].mean()
            erro_externo = np.abs(corrigida - limpa)[~manchas].mean()
            print(f"Erro absoluto médio nas manchas: {erro_antes:.6f} para {erro_depois:.6f}. "
                  f"Erro fora das manchas: {erro_externo:.6f}.", flush=True)
            self.assertLess(erro_depois, erro_antes)
            self.assertLess(erro_externo, erro_antes / 3)
            rel = json.loads((saida / "medidas.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rel["trechos"]["sequência"]["percentuais"]["soma"]), 5)
            self.assertTrue(np.isfinite(rel["detalhe_local"]["soma"]["fator"]))
            self.assertTrue((saida / "registro4.json").is_file())
            self.assertTrue((saida / "relatorio.md").is_file())

            medida_avulsa = pasta / "medida_avulsa"
            padrao_delta = saida / "soma" / "delta_%04d.exr"
            self.cli("measure", *quadros, "--delta", padrao_delta, "--saida", medida_avulsa,
                     "--threads", 1)
            avulsa = json.loads((medida_avulsa / "medidas.json").read_text(encoding="utf-8"))
            self.assertEqual(rel["trechos"]["sequência"]["percentuais"]["soma"],
                             avulsa["trechos"]["sequência"]["percentuais"]["corrigida"])

            integral = pasta / "forca_integral"
            self.cli("apply", *quadros, "--delta", padrao_delta, "--forca", 1,
                     "--saida", integral, "--formato", "png")
            for i, imagem in enumerate(corrigida, 1):
                obtido = cv2.imread(str(integral / f"quadro_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
                esperado = np.rint(imagem * 65535).astype(np.uint16)
                self.assertTrue(np.array_equal(obtido, esperado), f"Erro ao somar no quadro {i}.")

            decisao = pasta / "decisao"
            self.cli("decide", *quadros, "--execucao", saida, "--saida", decisao)
            dec = json.loads((decisao / "decisao.json").read_text(encoding="utf-8"))
            self.assertEqual(len(dec["passos"]), 4)
            usadas = dec["camadas_escolhidas"]
            self.assertTrue(usadas, dec["conclusao"])
            escolha = next(p for p in dec["passos"] if p["camadas"] == usadas)
            self.assertTrue(escolha["aceitavel"])
            self.assertGreater(escolha["reducao_tremor_%"], 0)
            melhor = max(p["reducao_tremor_%"] for p in dec["passos"] if p["aceitavel"])
            self.assertGreaterEqual(escolha["reducao_tremor_%"], melhor - dec["criterios"]["ganho_min_pontos"])
            for q in range(1, len(limpa) + 1):
                nome = f"delta_{q:04d}.exr"
                esperado = sum(ler_exr(saida / f"camada{k}" / nome) for k in usadas)
                self.assertTrue(np.allclose(ler_exr(decisao / "escolha" / nome), esperado, atol=1e-6))

            previa = pasta / "forca_zero"
            self.cli("apply", *quadros, "--delta", saida/"soma"/"delta_%04d.exr",
                     "--forca", 0, "--saida", previa, "--formato", "png")
            for q in range(1, len(limpa) + 1):
                a = cv2.imread(str(pasta / f"entrada_{q:04d}.png"), cv2.IMREAD_UNCHANGED)
                b = cv2.imread(str(previa / f"quadro_{q:04d}.png"), cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.array_equal(a, b), f"Força zero alterou o quadro {q}.")


if __name__ == "__main__":
    unittest.main()
