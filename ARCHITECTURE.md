# AlphaTrainer — arquitetura multimodelo

## Objetivo

O AlphaTrainer organiza motores de IA com arquiteturas diferentes numa plataforma comum para criação, treino, execução, avaliação, comparação e gestão. A camada multimodelo deve integrar os motores existentes sem forçar arquiteturas diferentes a implementar operações que não suportam.

O AlphaTrainer é independente da aplicação Alpha, embora modelos desenvolvidos no futuro possam vir a ser utilizados por outros produtos da Alpha Technologies.

## Componentes de gestão de modelos

- `models.factory.ModelFactory`: cria as arquiteturas disponibilizadas pela fábrica e mantém a compatibilidade com a função histórica `create_model`.
- `models.contracts`: define metadados e protocolos de capacidades.
- `models.adapters`: descreve as capacidades reconhecidas para as arquiteturas integradas.
- `models.manager.ModelManager`: gere modelos através do registo existente, incluindo operações de ciclo de vida como registo, consulta, renomeação, duplicação e remoção.
- `models.plugins.PluginRegistry`: permite registar explicitamente construtores confiáveis durante a execução. Não deve executar código arbitrário obtido de datasets, checkpoints ou ficheiros externos.
- `core.AlphaCore`: mantém a integração com as APIs legadas.

## Motores universais

### Treino — `training/universal.py`

`UniversalTrainingEngine` consulta o modelo e o adaptador. Encaminha redes densas para `training.trainer.Trainer`, Transformers para o respetivo método `fit()` e outros modelos compatíveis para um método `fit()` disponível. A compatibilidade depende das capacidades declaradas e dos métodos existentes.

### Avaliação — `evaluation/universal.py`

`UniversalEvaluationEngine` encaminha redes densas para `evaluate_model` e utiliza o método `evaluate()` nas arquiteturas que o disponibilizam. Devolve arquitetura, amostras, métricas e detalhes da avaliação.

### Benchmark — `evaluation/benchmark.py`

`UniversalBenchmarkEngine` avalia os modelos fornecidos e cria uma classificação segundo uma métrica numérica comum. Exige pelo menos dois modelos e que a métrica escolhida esteja disponível e seja válida em todos os resultados considerados.

### Otimização — `optimization/universal.py`

`UniversalOptimizationEngine` experimenta configurações de treino, avalia os resultados e produz um relatório comparativo. As métricas e configurações aceites dependem da arquitetura. A otimização deve preservar os pesos do modelo original e limpar os modelos temporários mesmo quando uma experiência falha.

### Runtime — `ai_runtime/runtime.py`

`AIRuntime` encaminha operações `predict`, `generate` e `forward` para os métodos suportados pelo modelo. Uma operação não é considerada suportada apenas por existir no runtime: as capacidades e os métodos do modelo têm de a permitir.

## Arquiteturas e limites

- As redes densas e o Transformer de linguagem são as arquiteturas principais reconhecidas pelos adaptadores atuais.
- Modelos personalizados podem ser registados, mas não recebem capacidades presumidas.
- CNN, processamento de imagem, diálogo avançado e aprendizagem por reforço não devem ser anunciados como motores implementados sem código e testes específicos.
- Os treinadores especializados continuam a executar o trabalho próprio de cada arquitetura; a camada universal encaminha operações, não torna os algoritmos internamente idênticos.
- A implementação do Transformer tem limitações e não deve ser descrita como equivalente a um LLM moderno de grande escala.

## Persistência versionada

`persistence/manager.py` integra os checkpoints Dense e Transformer através de `save_any` e `load_any`. O manifesto `alpha_persistence_v2` mantém a lista de modelos e datasets e referencia ficheiros versionados por geração. O manifesto só é substituído depois de os novos ficheiros estarem escritos; uma falha durante a gravação não publica a geração incompleta.

A leitura continua a aceitar o manifesto legado `alpha_persistence_v1`, usando os nomes de ficheiro históricos. Os datasets mantêm o formato `alpha_dataset_v1`.

`models.manager.ModelManager` exporta metadados no formato `alpha_model_metadata_v1`, incluindo identidade, arquitetura, configuração, capacidades inferidas, estado de treino, métricas, checkpoints e datas. Na recuperação, a arquitetura e as capacidades são comparadas com o modelo efetivamente carregado; metadados inválidos são rejeitados sem descartar um modelo válido.

As arquiteturas Dense e Transformer têm persistência suportada. Objetos personalizados sem checkpoint próprio são ignorados com aviso: o sistema não serializa nem executa código Python arbitrário. Ficheiros externos são tratados como dados não confiáveis e os nomes de ficheiros do manifesto V2 são validados para impedir caminhos fora das pastas de persistência.

## Segurança e compatibilidade

- Plugins são construtores Python registados explicitamente por código confiável; não devem ser carregados automaticamente a partir de ficheiros externos.
- Checkpoints, datasets e configurações importadas devem ser validados antes de serem utilizados.
- Ficheiros `.env`, credenciais, configurações Firebase privadas e segredos de serviços não devem ser distribuídos nem incluídos em repositórios públicos ou APKs.
- Alterações aos formatos persistidos devem considerar compatibilidade com os modelos e checkpoints já existentes.

## Fase 4 — Persistência e Gestão Multimodelo

**Objetivo:** tornar o ciclo de vida multimodelo recuperável entre execuções, sem quebrar as APIs e os formatos legados.

### Etapas executadas

1. Auditoria dos módulos `persistence/`, `checkpoints/`, `models/manager.py` e `registry/registry.py`.
2. Definição do formato `alpha_model_metadata_v1`, com identidade, arquitetura, capacidades, configuração e estado de treino.
3. Integração com manifesto `alpha_persistence_v2`, nomes de ficheiros versionados, escrita atómica e validação dos caminhos.
4. Recuperação de modelos Dense e Transformer, mantendo a leitura de manifestos V1 e datasets legados.
5. Testes de reinício, metadados, Transformer, duplicação, renomeação, remoção, corrupção, caminhos inválidos e falhas de gravação.
6. Atualização desta documentação e do README.

### Resultado da validação isolada

- **440 testes aprovados** na cópia de validação do ZIP.
- `compileall` concluído sem erros de compilação.
- Compatibilidade V1 e persistência V2 cobertas por testes.
- A execução final no Termux/Android de destino continua pendente.

**Estado:** implementação preparada; a fase só deve ser declarada concluída depois da validação no dispositivo do utilizador.
