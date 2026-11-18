# Notas do laboratório

## Estado inicial

- O laboratório foi criado separado do código de produção.
- O checkout atual não contém os três CSVs originais da auditoria.
- A execução real depende de `FA_CLIP`, `FA_OBJECT_MODEL` e `FA_PITCH_MODEL`.
- Sem esses artefatos, os notebooks devem registrar pré-requisitos ausentes e
  parar a análise daquele estágio sem inventar resultados.

## Decisões

- A transformação baseline é imagem → coordenadas métricas do campo.
- O ponto fixo padrão é o centro do campo projetado na imagem e depois
  reprojetado no campo; variantes podem declarar uma definição diferente.
- Nenhum limiar experimental é considerado validado sem sweep e comparação
  com o salto residual ou ground truth.

