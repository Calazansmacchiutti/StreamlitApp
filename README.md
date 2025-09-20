# 🌊 Sistema de Monitoramento de Estações de Tratamento

Um sistema completo de monitoramento em tempo real para estações de tratamento de água, desenvolvido com Streamlit e tecnologias de análise de dados avançadas.

## 🚀 Características Principais

### 📊 Dashboard Interativo
- **Visão Geral**: KPIs em tempo real, status das estações, alertas ativos
- **Análise de Séries Temporais**: Decomposição sazonal, ARIMA, Prophet, análise de tendências
- **Previsões**: Modelos de ML (Random Forest, XGBoost, Prophet, ARIMA)
- **Detecção de Anomalias**: Isolation Forest, DBSCAN, One-Class SVM, Z-Score, IQR
- **Relatórios**: Geração automática de relatórios em PDF, Excel, HTML
- **Configurações**: Interface completa para configuração do sistema

### 🏭 Estações Configuradas
- **Two Mouths**: Estação principal (pH, turbidez, cloro, vazão, temperatura)
- **New Rose of Rocky**: Estação secundária (pH, DO, BOD, COD, TSS)
- **Botanic Garden**: Estação terciária (pH, nitrogênio, fósforo, coliformes, turbidez)

### 🔧 Funcionalidades Técnicas
- **Monitoramento em Tempo Real**: Atualização automática de dados
- **Alertas Inteligentes**: Sistema de notificações por email, Slack, Teams
- **Análise Preditiva**: Previsões com intervalos de confiança
- **Detecção de Anomalias**: Múltiplos algoritmos de detecção
- **Validação Cruzada**: Avaliação robusta de modelos
- **Exportação de Dados**: Múltiplos formatos (CSV, Excel, PDF, JSON)

## 📁 Estrutura do Projeto

```
StreamlitApp/
├── app.py                 # Aplicação principal
├── pages/                 # Páginas do dashboard
│   ├── 1_📊_Overview.py
│   ├── 2_📈_Time_Series.py
│   ├── 3_🔮_Predictions.py
│   ├── 4_⚠️_Anomalies.py
│   ├── 5_📑_Reports.py
│   └── 6_⚙️_Settings.py
├── modules/               # Módulos de funcionalidades
│   ├── data_loader.py     # Carregamento de dados
│   ├── preprocessor.py    # Pré-processamento
│   ├── time_series.py     # Análise de séries temporais
│   ├── anomaly_detection.py # Detecção de anomalias
│   ├── visualizations.py  # Visualizações
│   └── alerts.py          # Sistema de alertas
├── data/                  # Dados das estações
│   ├── Two Mouths.csv
│   ├── New Rose of Rocky.csv
│   └── Botanic Garden.csv
├── config/                # Configurações
│   └── config.yaml
├── requirements.txt       # Dependências
├── generate_synthetic_data.py # Gerador de dados sintéticos
└── README.md
```

## 🛠️ Instalação

### 1. Clone o repositório
```bash
git clone <repository-url>
cd StreamlitApp
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute a aplicação
```bash
streamlit run app.py
```

## 📊 Dados Sintéticos

O projeto inclui um gerador de dados sintéticos realistas para demonstração:

```bash
python generate_synthetic_data.py
```

### Características dos Dados
- **Período**: 90 dias de dados históricos
- **Frequência**: Dados a cada 15 minutos
- **Parâmetros**: Específicos para cada estação
- **Padrões**: Sazonais, diários e tendências realistas
- **Anomalias**: 1% de outliers para testes
- **Valores Faltantes**: 0.5% para simular condições reais

## 🔧 Configuração

### Arquivo de Configuração (config/config.yaml)
- **Estações**: Configuração de estações e parâmetros
- **Limites**: Limites de controle por parâmetro
- **Alertas**: Configuração de regras e notificações
- **Previsões**: Configuração de modelos preditivos
- **Sistema**: Configurações gerais do sistema

### Personalização
1. Acesse a página "⚙️ Settings"
2. Configure estações, parâmetros e limites
3. Ajuste regras de alertas e notificações
4. Configure modelos de previsão
5. Salve as configurações

## 📈 Modelos de Previsão

### Modelos Disponíveis
- **Prophet**: Para séries temporais com sazonalidade
- **ARIMA**: Para séries estacionárias
- **Random Forest**: Para padrões não-lineares
- **XGBoost**: Para alta performance
- **LSTM**: Para padrões complexos (em desenvolvimento)

### Métricas de Avaliação
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

## ⚠️ Detecção de Anomalias

### Algoritmos Implementados
- **Isolation Forest**: Para outliers multivariados
- **DBSCAN**: Para clusters de anomalias
- **One-Class SVM**: Para detecção de novidades
- **Z-Score**: Para outliers estatísticos
- **IQR**: Para outliers baseados em quartis

### Classificação de Severidade
- **Crítico**: Valores fora dos limites críticos
- **Alto**: Valores fora dos limites normais
- **Médio**: Anomalias detectadas por algoritmos
- **Baixo**: Variações menores

## 📧 Sistema de Notificações

### Métodos Suportados
- **Email**: SMTP com templates HTML
- **Slack**: Webhooks com formatação rica
- **Microsoft Teams**: Cards interativos
- **Webhook**: Para integração com sistemas externos

### Configuração de Alertas
- **Severidade Mínima**: Filtrar alertas por nível
- **Frequência**: Controlar spam de notificações
- **Período Silencioso**: Evitar alertas em horários específicos
- **Escalação**: Alertas automáticos para supervisores

## 📑 Relatórios

### Tipos de Relatórios
- **Operacional**: Para operadores de campo
- **Executivo**: Para gestão e direção
- **Conformidade**: Para órgãos reguladores
- **Manutenção**: Para equipes técnicas

### Formatos Suportados
- **PDF**: Para impressão e arquivo
- **Excel**: Para análise detalhada
- **HTML**: Para visualização web
- **Word**: Para documentos corporativos

## 🔍 Análise de Séries Temporais

### Funcionalidades
- **Decomposição Sazonal**: Tendência, sazonalidade e resíduos
- **Análise de Estacionariedade**: Testes ADF e KPSS
- **Análise Espectral**: Identificação de frequências dominantes
- **Análise de Tendências**: Regressão linear e médias móveis

### Visualizações
- **Gráficos Interativos**: Plotly com zoom e hover
- **Múltiplas Séries**: Comparação de parâmetros
- **Limites de Controle**: Visualização de limites
- **Anomalias**: Marcadores de eventos anômalos

## 🚀 Executando o Sistema

### Modo Desenvolvimento
```bash
streamlit run app.py --server.port 8501
```

### Modo Produção
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Variáveis de Ambiente
```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
```

## 📊 Monitoramento de Performance

### Métricas do Sistema
- **Uptime**: Disponibilidade do sistema
- **Memória**: Uso de RAM
- **CPU**: Utilização do processador
- **Armazenamento**: Espaço em disco

### Logs
- **Níveis**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Rotação**: Logs automáticos por data
- **Filtros**: Por nível e período
- **Exportação**: Download de logs

## 🔒 Segurança

### Autenticação
- **Login**: Sistema de autenticação (opcional)
- **Sessões**: Timeout configurável
- **Tentativas**: Limite de tentativas de login
- **Auditoria**: Log de ações dos usuários

### Dados
- **Backup**: Backup automático de configurações
- **Criptografia**: Dados sensíveis criptografados
- **Acesso**: Controle de acesso por usuário

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

### Padrões de Código
- **PEP 8**: Seguir padrões Python
- **Docstrings**: Documentar funções e classes
- **Type Hints**: Usar anotações de tipo
- **Testes**: Adicionar testes para novas funcionalidades

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

### Documentação
- **README**: Este arquivo
- **Código**: Comentários e docstrings
- **Configuração**: Arquivo config.yaml

### Contato
- **Issues**: Use o sistema de issues do GitHub
- **Email**: [seu-email@exemplo.com]
- **Documentação**: [link-para-docs]

## 🎯 Roadmap

### Próximas Funcionalidades
- [ ] **LSTM Models**: Implementação completa de redes neurais
- [ ] **Ensemble Methods**: Combinação de múltiplos modelos
- [ ] **Real-time Data**: Integração com sensores IoT
- [ ] **Mobile App**: Aplicativo móvel para monitoramento
- [ ] **API REST**: API para integração com outros sistemas
- [ ] **Machine Learning Pipeline**: Pipeline automatizado de ML
- [ ] **Advanced Analytics**: Análises estatísticas avançadas
- [ ] **Multi-language**: Suporte a múltiplos idiomas

### Melhorias Planejadas
- [ ] **Performance**: Otimização de consultas e visualizações
- [ ] **UI/UX**: Melhorias na interface do usuário
- [ ] **Testing**: Cobertura de testes mais abrangente
- [ ] **Documentation**: Documentação mais detalhada
- [ ] **Deployment**: Scripts de deploy automatizados

---

**Desenvolvido com ❤️ usando Streamlit e tecnologias de análise de dados**

*Sistema de Monitoramento v2.0 - 2024*