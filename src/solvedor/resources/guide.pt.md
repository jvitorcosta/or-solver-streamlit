### :material/help: Como escrever um problema

Um problema tem até três partes. Só o objetivo é obrigatório. As palavras-chave são as do
inglês (ou `maximizar`, `minimizar`, `sujeito a` e `onde`).

```text
maximize 3x + 2y          # ou max, maximizar, minimize, min, minimizar

subject to:               # ou s.t. / st / sujeito a; os dois-pontos são opcionais
    x + 2y <= 4           # <=, >=, = (≤ e ≥ também funcionam)
    2x + y - z <= 4

where:                    # ou onde; opcional: limites e tipos
    x <= 3                # um limite superior
    0 <= y <= 10          # os dois limites
    free z                # z pode ser negativo
    integer x             # ou binary
```

- **As variáveis são não negativas**, a menos que você escreva `free` ou dê um limite inferior.
- Coeficientes podem ser `3x`, `3*x` ou `3 x`, com ponto decimal (`0.5x`) ou expoente (`1e3x`).
- Constantes e variáveis podem ficar dos dois lados de uma restrição: `x + 5 >= y` funciona.
- Nomes de variáveis começam com letra ou `_` e podem ter dígitos (`produto_a`, `x1`).
- `#` inicia um comentário.
- Se algo estiver errado, a mensagem de erro indica a linha.

### :material/function: Resultados

- **Ótima**: a melhor solução, com prova.
- **Melhor solução antes do limite de tempo**: um problema inteiro parou após 10 segundos com
  uma boa solução, sem prova de que é a melhor.
- **Inviável**: nenhum ponto satisfaz todas as restrições.
- **Ilimitado**: o objetivo pode melhorar para sempre; falta algum limite.

Problemas lineares são resolvidos com o GLOP e problemas inteiros com o SCIP, ambos do Google
OR-Tools.

### :material/show_chart: O gráfico (problemas com duas variáveis)

- **Área verde**: a região viável, calculada a partir das suas restrições e limites.
- **Linhas tracejadas**: uma por restrição, com a própria restrição como rótulo.
- **Seta**: a direção em que o objetivo melhora.
- **Estrela**: a solução. Num problema linear ela fica num vértice da região.

Em problemas inteiros o gráfico mostra a região contínua; a solução é um ponto de coordenadas
inteiras dentro dela.
