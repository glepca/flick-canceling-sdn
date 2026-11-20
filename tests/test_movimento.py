"""Dois vizinhos confirmam o pulso, mas vetam a correção num disco que gira."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from flick.movimento import trava
from flick.fluxo import exportar
from flick.imagens import ler_cor


class Movimento(unittest.TestCase):
    def test_correspondencia_aparente_entre_texturas_diferentes(self):
        # Um fluxo que vai e volta perfeitamente ainda pode ligar facetas distintas.
        rng = np.random.default_rng(42)
        p = rng.uniform(.2, .4, (80, 128)).astype(np.float32)
        a = np.repeat(p[..., None], 3, axis=2)
        b = a + .12
        b[:, 64:] = np.repeat((.72 - p[:, 64:])[..., None], 3, axis=2)
        pedido = np.full_like(a, -.12)

        class FluxoNulo:
            def calc(self, origem, destino, inicial):
                return np.zeros((*origem.shape, 2), np.float32)

        with patch("flick.movimento.cv2.DISOpticalFlow_create", return_value=FluxoNulo()):
            seq = [a, b, a]
            ganho, _ = trava(seq, None, 1, pedido, 4., .70, .04)
        d = np.abs(ganho * pedido)
        self.assertEqual(d[12:-12, 80:-12].max(), 0)
        self.assertAlmostEqual(float(d[12:-12, 12:50].mean()), .04, places=5)
        self.assertLessEqual(d.max(), .040001)

    def test_entrega_recortada_com_soma_e_mascara(self):
        with tempfile.TemporaryDirectory() as temporario:
            pasta = Path(temporario)
            rng = np.random.default_rng(6)
            placa = rng.uniform(.2, .7, (48, 64, 3)).astype(np.float32)
            base = np.stack([placa, placa + .1, placa])
            np.save(pasta/"entrada.npy", base)
            regiao = np.ones(base.shape[:3], np.float32)
            regiao[:, :10, :] = 0
            np.save(pasta/"regiao.npy", regiao)
            partes = []
            for n in range(4):
                arquivo = pasta/f"parte{n}.npy"
                np.save(arquivo, np.full_like(base, -.03, dtype=np.float16))
                partes.append(arquivo)
            exportar(partes, pasta, (64, 48), (0, 0, 64, 48), (64, 48), 20,
                     "half", range(1, 2), 4., pasta/"entrada.npy", pasta/"regiao.npy")
            nome = "delta_0021.exr"
            self.assertEqual([f.name for f in (pasta/"soma").iterdir()], [nome])
            total = ler_cor(pasta/"soma"/nome)
            camadas = [ler_cor(pasta/f"camada{n}"/nome).astype(np.float32) for n in range(1, 5)]
            np.testing.assert_array_equal(total, sum(camadas).astype(np.float16))
            acao = cv2.imread(str(pasta/"acao"/"acao_0021.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(acao > 0, np.any(np.stack(camadas) != 0, axis=(0, 3)))
            self.assertFalse(acao[:10].any())
            self.assertGreater(acao[16:40, 8:48].mean(), 250.)
            self.assertEqual(list(json.loads((pasta/"trava_fluxo.json").read_text())["quadros"]), ["21"])

    def test_pulso_estatico_rotacao_e_oclusao(self):
        rng = np.random.default_rng(7)
        h, w = 192, 256
        y, x = np.indices((h, w))
        centro = (168, 96)
        raio = np.hypot(x - centro[0], y - centro[1])
        textura = cv2.GaussianBlur(rng.uniform(.10, .80, (h, w)).astype(np.float32),
                                   (0, 0), .8)
        quadros = []
        for t in (-1, 0, 1):
            rotacao = cv2.getRotationMatrix2D(centro, t * 9., 1.)
            movel = cv2.warpAffine(textura, rotacao, (w, h))
            placa = np.where(raio < 70, movel, textura)
            if t == 0:
                placa += .12
            quadros.append(np.repeat(placa[..., None], 3, axis=2))
        regiao = np.ones((3, h, w), np.float32)
        regiao[2, 52:75, 42:72] = 0
        pedido = np.full_like(quadros[0], -.12)
        ganho, info = trava(quadros, regiao, 1, pedido, 4.)
        self.assertGreater(ganho[92:136, 24:60].mean(), .95)
        self.assertEqual(ganho[56:71, 47:64].max(), 0)
        self.assertLess(ganho[(raio > 43) & (raio < 59)].mean(), .015)
        self.assertGreater(info["pixels_velozes"], 3000)
        ganho, _ = trava(quadros, regiao, 1, pedido, 4., .70, .04)
        self.assertLess(ganho[(raio > 43) & (raio < 59)].mean(), .015)
        self.assertGreater(ganho[92:136, 24:60].mean(), .32)

        # Conservar uma variação monotônica, mesmo com grande delta candidato.
        base = quadros[0]
        ganho, _ = trava([base - .02, base, base + .02], regiao, 1, pedido, 4.)
        self.assertEqual(ganho[92:136, 24:60].max(), 0)


if __name__ == "__main__":
    unittest.main()
