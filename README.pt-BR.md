# Flick Canceling SDÑ

> Gambiarra também é parte da criatividade.

[English](README.md) · **Português**

Feita pela [Smart Diseños (SDÑ)](https://smartdisenos.com), *Non-Traditional Creative Studio*, no meio de um filme 3D de verdade, quando a água e o gelo do render começaram a piscar e não havia tempo de renderizar de novo. A gente divulga para quem passar pelo mesmo aperto.

Ferramenta de estúdio para reduzir manchas de luz e cor que aparecem e desaparecem em um plano. Entrega quatro deltas RGB com sinal, a soma e uma máscara PNG dos pixels em que alguma camada atuou. A intensidade pode ser decidida na composição.

## Instalação e uso

Use Python 3.14 em um ambiente virtual. Numere os quadros sem lacunas, com a mesma resolução e curva de cor em todo o plano. A receita foi desenhada para RGB em escala de 0 a 1. PNG e TIFF aceitam 8 ou 16 bits. EXR usa os canais principais R, G e B. A ferramenta trabalha nos valores recebidos, sem transformação de cor.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/flick run --entrada 'quadros/luz_%04d.png' --inicio 1 --fim 48 \
  --regiao 'mascaras/objeto_%04d.png' --excluir mascaras/proteger.png \
  --receita flick/receita.json --processos 4 --saida saida/plano01
.venv/bin/flick decide --entrada 'quadros/luz_%04d.png' --inicio 1 --fim 48 \
  --execucao saida/plano01 --saida saida/decisao01
.venv/bin/flick apply --entrada 'quadros/luz_%04d.png' --inicio 1 --fim 48 \
  --delta 'saida/decisao01/escolha/delta_%04d.exr' --forca 1 --saida saida/previa01
.venv/bin/flick measure --entrada 'quadros/luz_%04d.png' --inicio 1 --fim 48 \
  --corrigida 'saida/previa01/quadro_%04d.png' --regiao 'mascaras/objeto_%04d.png' \
  --trecho 'meio=10,38,2' --saida saida/medida01
.venv/bin/python -m unittest discover -s tests
```

Cada comando de entrega pede uma pasta inédita. O resultado da prévia é `clip(original + força × delta, 0, 1)`. Força zero com o mesmo formato copia o original. O EXR de delta tem sinal e usa half de 16 bits por padrão. Use `--precisao float32` para gravar a soma e a redução espacial com maior precisão. O processamento usa float32 e mantém os arredondamentos em half nas gravações intermediárias da receita de referência.

## Deixe a ferramenta decidir

Ninguém deveria julgar cada plano no olho. O `flick decide` revisa o próprio trabalho: aplica as camadas em ordem (1, 1+2, 1+2+3, as quatro) e mede tremor e detalhe só onde a ferramenta atuou. Uma combinação é vetada quando mantém menos de 90% do detalhe (`--detalhe-min`) ou quando mais de 1% dos quadradinhos de 64 pixels ganha detalhe demais, sinal de borda dobrada (`--fantasma-max`). Entre as que sobram, fica a menor combinação a até 2 pontos da melhor (`--ganho-min`): camada que quase não ajuda não entra. As camadas não são monotônicas (uma pode quase não agir e a seguinte agir muito), por isso todas são medidas antes da decisão. O delta escolhido vai para `escolha/` e o porquê, para `decisao.json`.

Num render 3D com gelo piscando, ficaram as quatro camadas: 30% menos tremor, 93% do detalhe. Num timelapse em domínio público ([Growing grass, flicker not removed](https://commons.wikimedia.org/wiki/File:Growing_grass_(flicker_not_removed).ogv)), ficaram as camadas 1 a 3: 55% menos tremor sem perder detalhe, contra 28% de um deflicker simples de brilho global.

## Receita e máscaras

As duas primeiras camadas estabilizam quadradinhos por registro afim ECC, escolhem pela luminância uma referência de aparência com memória e protegem movimento consistente. A segunda desloca a grade sobre o resultado da primeira. A terceira detecta alternância em relação à média dos dois vizinhos. A quarta acompanha máscaras manuais e regiões automáticas, suavizando a trajetória de cor em cada região estabilizada.

- Os padrões estão em `flick/receita.json`. Um JSON parcial substitui só os campos fornecidos. Na CLI, por exemplo, `--param camada2.sigma_cor=4.0` ou `--param camada4.automatica=false`. Distâncias multiplicam pela largura da tela dividida por 3840, áreas pelo quadrado dessa razão. Os tamanhos de janela temporal são em quadros. A dilatação e a suavização da região usam a largura nativa da máscara dividida por 1920.
- `--regiao` aceita PNG fixo ou por quadro, em tons de cinza. Branco permite a correção, preto a exclui. O envelope da região pode avançar sobre a borda segundo a receita. `--excluir` protege cada pixel marcado, inclusive na camada manual. A máscara manual tem envelope próprio.
- Para Cryptomatte, use `--cryptomatte 'passes/matte_%04d.exr' --camada-crypto CryptoObject --id-crypto Vidro`. Repita a flag para somar coberturas. Aceita nomes do manifesto incorporado ou o ID hexadecimal de oito dígitos. A região usa a soma dos pesos de cada par ID/cobertura.
- Para acompanhar um objeto, use `--mascara-track '{"arquivo":"mascara.png","quadro":24}'`. Também aceita JSON com `{"pontos":[[x1,y1],[x2,y2],[x3,y3]]}` usando números. A chave opcional `janela` recebe `[x,y,largura,altura]` para delimitar o registro. Coordenadas partem do alto à esquerda da tela inteira. Caminhos em receitas são relativos ao JSON. A referência automática padrão é o meio da sequência.
- `--faixa X Y L A` processa um recorte e escreve zero fora dele. `--tamanho-saida L A` define a resolução de entrega. O adaptador NPY aceita `[tempo,altura,largura,BGR]`, inteiros sem sinal ou floats normalizados. Para um NPY já recortado, declare `--tela L A` e a faixa. `--origem-npy` identifica o primeiro quadro armazenado ao extrair um intervalo.
- `--entregar Q0 Q1` restringe o intervalo exportado. As camadas usam todos os quadros de `--inicio` a `--fim` como contexto.
- `--trava-fluxo PX` acrescenta uma proteção para movimento forte na exportação. O fluxo óptico procura o mesmo ponto nos dois quadros vizinhos e verifica a volta ao quadro atual. O peso cai linearmente entre metade de PX e PX pixels de deslocamento por quadro, medidos na resolução da entrada. O pixel perde a correção ao sair da região em um dos vizinhos, ao falhar a volta ou ao atingir o limite de velocidade. Em cada canal, só permanece a parcela do delta que aproxima o centro dos dois vizinhos na mesma direção. Os extremos da sequência de contexto ficam com delta zero. `trava_fluxo.json` informa a cobertura por quadro. A receita original continua sendo o padrão.
- `--trava-textura 0.70`, junto com a trava de fluxo, compara a luminância em janelas de 15 × 15 pixels nos dois vizinhos já alinhados. Uma correlação local até esse limiar veta a ação. Acima dele, o peso cresce linearmente até 1. O controle de contraste também atenua janelas cujo desvio padrão fica abaixo de 0,008 em valores de código normalizados.
- `--teto-delta 0.04`, também opcional junto com a trava de fluxo, comprime gradualmente amplitudes acima de metade do teto. O maior deslocamento de canal se aproxima do teto e os três canais recebem o mesmo fator. A quantização de cada camada acontece depois dessa operação.

## Medidas

`medidas.json` e `medidas.md` trazem as reduções de pulo total, pulo largo, vermelho e troca, a variação de detalhe e o maior aumento local de detalhe. Cada centro usa dois vizinhos anteriores e dois posteriores. O maior aumento local cobre o quadro inteiro. Os arquivos `registro1.json` a `registro4.json` guardam os diagnósticos de seleção e correlação do registro. `execucao.json` registra tempos e, no macOS, picos amostrados de memória da árvore de processos. Os caches mapeados em disco ficam na pasta da execução.

Antecedentes em correção localizada de flicker incluem Pitié et al. (2006) e o Furnace (2011), da Foundry. [Artigo no CVMP](https://doi.org/10.1049/cp:20061935) · [Manual do Furnace 4.2, p. 79](https://thefoundry.s3.amazonaws.com/products/furnace/nuke/documentation/Furnace_4.2_Nuke.pdf#page=79)

Licença Apache 2.0 em `LICENSE`.
