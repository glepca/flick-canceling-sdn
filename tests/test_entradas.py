"""Conferências dos formatos, da geometria e das máscaras em dados gerados."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

from flick.cli import padroes
from flick.fluxo import projetar
from flick.imagens import (Sequencia, crypto, gravar_exr, leitor_mascara, ler_cor, nome,
                           preparar_mascaras)


class Entradas(unittest.TestCase):
    def setUp(self):
        temporaria = tempfile.TemporaryDirectory()
        self.addCleanup(temporaria.cleanup)
        self.p = Path(temporaria.name)

    def test_cores_arquivos(self):
        numeros = np.arange(90, dtype=np.uint16).reshape(5, 6, 3)*700
        for ext, dt in (("png", np.uint8), ("png", np.uint16), ("tiff", np.uint16)):
            arq = self.p/f"imagem.{ext}"
            img = numeros.astype(dt)
            self.assertTrue(cv2.imwrite(str(arq), img))
            np.testing.assert_array_equal(ler_cor(arq), img)
        cores = numeros.astype(np.float32) / 15000 - 1.25
        for modo, dt in (("half", np.float16), ("float32", np.float32)):
            arq = self.p/f"{modo}.exr"
            gravar_exr(arq, cores, modo)
            np.testing.assert_array_equal(ler_cor(arq), cores.astype(dt))

    def test_cripto(self):
        cob = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
        bits = np.array(0x12345678, np.uint32).view(np.float32)
        canais = {"Objeto00.R": np.full_like(cob, bits), "Objeto00.G": cob,
                  "Objeto00.B": np.zeros_like(cob), "Objeto00.A": 1-cob}
        cab = {"cryptomatte/1/name":"Objeto", "cryptomatte/1/manifest":'{"Vidro":"12345678"}'}
        arq = self.p/"matte.exr"
        with OpenEXR.File(cab, canais) as f:
            f.write(str(arq))
        for ids in (["Vidro"], ["12345678"], ["0x12345678"]):
            np.testing.assert_array_equal(crypto(arq, ids, None), cob)
        with self.assertRaisesRegex(ValueError, "ausente"):
            crypto(arq, ["metal"], "Objeto")

    def test_mascaras_e_faixa(self):
        padrao = self.p/"reg_%02d.png"
        negativa = self.p/"neg.png"
        exc = np.zeros((12, 16), np.uint8)
        exc[:, 4:7] = 255
        cv2.imwrite(str(negativa), exc)
        for q in (12, 13):
            r = np.zeros_like(exc)
            r[q-9:, :q] = 255
            cv2.imwrite(str(nome(padrao, q)), r)
        faixa = (2, 6, 12, 6)
        reg, neg = preparar_mascaras(leitor_mascara(padrao,12), [leitor_mascara(negativa,12)],
                                     (16,12), faixa, 2, self.p, {"dilatacao":1,"suavizacao":0})
        real = np.load(reg)
        self.assertTrue(real.any())
        self.assertFalse(real[:, :, 2:5].any())
        self.assertTrue(np.load(neg)[:, :, 2:5].all())
        self.assertFalse(np.array_equal(real[0], real[1]))
        d = np.ones((6,12,3), np.float32)
        tela = projetar(d, (16,12), faixa, (8,6))
        self.assertEqual(tela.sum(), 3*6*3)
        vazio = tela.copy()
        vazio[3:, 1:7] = 0
        self.assertFalse(vazio.any())

    def test_npy_e_receita(self):
        p = self.p/"volume.npy"
        f = np.arange(7*12*16*3, dtype=np.uint16).reshape(7,12,16,3)
        np.save(p,f)
        s = Sequencia(p,6,10,4)
        np.testing.assert_array_equal(s.quadro(4), f[6])
        with self.assertRaises(ValueError):
            Sequencia(p, 4, 11, 4)
        r = Path(__file__).resolve().parents[1]/"flick"/"receita.json"
        self.assertEqual(json.loads(r.read_text()), padroes())
