"""Confere o sinal e a escala das cinco medidas em uma pulsação sem movimento."""

import unittest

import numpy as np

from flick.medidas import medir


class TesteDasMedidas(unittest.TestCase):
    def test_pulso_colorido(self):
        altura, largura = 48, 64
        y, x = np.mgrid[:altura, :largura].astype(np.float32)
        base = np.full((altura, largura, 3), 0.30, np.float32)
        base[..., 1] += 0.08 * np.sin(x / 2) * np.cos(y / 3)
        antes = np.repeat(base[None], 5, axis=0)
        depois = antes.copy()
        antes[2, :, :, 0] += 0.20
        antes[2, :, :, 2] += 0.10
        depois[2, :, :, 0] += 0.10
        # A região interior evita que a borda do remapeamento afete a medição.
        mascara = np.zeros((altura, largura), dtype=np.float32)
        mascara[8:-8, 8:-8] = 1
        medida = medir(antes.__getitem__, {"corrigida": depois.__getitem__},
                       lambda i: mascara, {"pulso": [2]}, len(antes))
        valores = medida["trechos"]["pulso"]["percentuais"]["corrigida"]
        reducao_luma = round(100 * (1 - 0.10 * 0.0722 / (0.20 * 0.0722 + 0.10 * 0.2126)), 1)
        self.assertEqual(valores, [reducao_luma, reducao_luma, 100., reducao_luma, 0.])

    def test_mascara_vazia_e_referencia_nula(self):
        preto = np.zeros((16, 16, 3), dtype=np.float32)
        vazia = np.zeros((16, 16), dtype=np.float32)
        for regiao, centros in ((None, [2]), (lambda i: vazia, [])):
            medida = medir(lambda i: preto, {"identidade": lambda i: preto}, regiao, {"pulso": [2]}, 5)
            dados = medida["trechos"]["pulso"]
            self.assertEqual(dados["indices_com_regiao"], centros)
            self.assertEqual(dados["percentuais"]["identidade"], [None] * 5)
