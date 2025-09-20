# 🌊 Sistema de Monitoramento Avançado de Estações de Tratamento de Água

Este projeto implementa um "Mega Dashboard" profissional e completo usando Streamlit para monitorar estações de tratamento de água. Ele oferece uma arquitetura modular robusta, visualizações avançadas, análise de séries temporais, detecção de anomalias, sistema de alertas, API REST, backup automático e auditoria completa.

## ✨ Funcionalidades Principais

### 📊 Dashboard Interativo
- **Visão Geral:** KPIs principais, monitoramento em tempo real e matriz de correlação
- **Análise de Séries Temporais:** Decomposição sazonal, modelos ARIMA/SARIMA, Prophet, LSTM
- **Análise Detalhada:** Estatísticas descritivas, análise de distribuição e tendências
- **Previsões Avançadas:** Ensemble de modelos (Prophet, Random Forest, XGBoost, SARIMA, LSTM)
- **Alertas e Anomalias:** Múltiplos algoritmos (Isolation Forest, One-Class SVM, DBSCAN, Z-Score, IQR)
- **Relatórios:** Geração automática em PDF, Excel, Word, PowerPoint e HTML

### 🔧 Funcionalidades Avançadas
- **API REST Completa:** Endpoints para integração com outros sistemas
- **Sistema de Backup:** Backup automático com criptografia e compressão
- **Auditoria Completa:** Log de todas as ações do sistema
- **Monitoramento de Sistema:** Métricas de CPU, memória, disco e rede
- **Cache Inteligente:** Sistema de cache para otimização de performance
- **Detecção de Drift:** Monitoramento de mudanças na distribuição dos dados
- **Features Engineering:** Criação automática de features temporais e de lag
- **Análise Multivariada:** Detecção de anomalias em múltiplas dimensões

## 🏗️ Arquitetura do Sistema

### Módulos Principais
```
modules/
├── data_loader.py          # Carregamento com cache e metadados
├── preprocessor.py         # Pré-processamento avançado
├── time_series.py          # Análise de séries temporais
├── anomaly_detection.py    # Detecção de anomalias
├── visualizations.py       # Visualizações interativas
├── alerts.py              # Sistema de alertas
├── api_server.py          # API REST
└── backup_audit.py        # Backup e auditoria
```

### Configuração Avançada
```
config/
├── config.yaml            # Configuração básica
└── config_advanced.yaml   # Configuração avançada
```

## 🚀 Instalação e Configuração

### 1. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 2. Geração de Dados Sintéticos
```bash
python generate_synthetic_data.py
```

### 3. Teste das Funcionalidades Avançadas
```bash
python test_advanced_features.py
```

### 4. Execução do Dashboard
```bash
streamlit run app.py
```

### 5. Execução da API REST
```bash
python -m modules.api_server
```

## 🔧 Funcionalidades Avançadas Detalhadas

### 📊 DataLoader Avançado
- **Cache Inteligente:** Sistema de cache com LRU para otimização
- **Metadados:** Informações detalhadas sobre estações e dados
- **Backup/Restore:** Funcionalidades de backup e restauração
- **Suporte a Banco de Dados:** Conexão com SQLite e outros SGBDs
- **Validação de Dados:** Verificação de qualidade e integridade

### 🧹 DataPreprocessor Avançado
- **Tratamento de Valores Ausentes:** Múltiplas estratégias (interpolação, KNN, etc.)
- **Features de Lag:** Criação automática de features temporais
- **Features de Janela Móvel:** Médias, desvios, quartis móveis
- **Features Temporais:** Extração de componentes cíclicos
- **Features de Interação:** Combinações entre variáveis
- **PCA:** Redução de dimensionalidade
- **Seleção de Features:** Métodos estatísticos para seleção
- **Detecção de Drift:** Monitoramento de mudanças na distribuição

### 📈 TimeSeriesAnalyzer Avançado
- **Testes de Estacionariedade:** ADF e KPSS
- **Auto ARIMA:** Seleção automática de parâmetros
- **SARIMA:** Modelos sazonais avançados
- **Prophet:** Modelos com sazonalidade complexa
- **Ensemble:** Combinação de múltiplos modelos
- **Decomposição STL:** Decomposição robusta de séries temporais
- **Detecção de Sazonalidade:** Identificação automática de períodos

### ⚠️ AnomalyDetector Avançado
- **Isolation Forest:** Para outliers multivariados
- **One-Class SVM:** Para detecção de novidades
- **DBSCAN:** Para clusters de anomalias
- **Métodos Estatísticos:** Z-Score, IQR, MAD
- **LSTM Autoencoder:** Para padrões complexos
- **Detecção Multivariada:** Mahalanobis, PCA
- **Ensemble:** Combinação de múltiplos métodos
- **Classificação de Severidade:** Crítico, Alto, Médio, Baixo

### 🚨 Sistema de Alertas Avançado
- **Múltiplos Canais:** Email, SMS, Slack, Teams, Webhook
- **Escalação Automática:** Baseada em severidade e tempo
- **Templates Personalizáveis:** HTML, Markdown
- **Rate Limiting:** Controle de frequência de alertas
- **Filtros Inteligentes:** Por estação, parâmetro, severidade
- **Histórico Completo:** Log de todos os alertas

### 🌐 API REST Completa
- **Endpoints Principais:**
  - `GET /api/v1/health` - Health check
  - `GET /api/v1/data` - Dados das estações
  - `GET /api/v1/alerts` - Alertas ativos
  - `POST /api/v1/analysis/anomalies` - Detecção de anomalias
  - `POST /api/v1/analysis/forecast` - Geração de previsões
  - `POST /api/v1/reports` - Geração de relatórios
  - `GET /api/v1/metrics` - Métricas do sistema

- **Autenticação:** JWT, API Key, OAuth2
- **Rate Limiting:** Controle de requisições
- **CORS:** Suporte a requisições cross-origin
- **Documentação:** Swagger/OpenAPI integrado

### 💾 Sistema de Backup e Auditoria
- **Backup Automático:** Diário, incremental, completo
- **Criptografia:** AES-256 para dados sensíveis
- **Compressão:** ZIP para otimização de espaço
- **Retenção:** Políticas configuráveis
- **Restauração:** Recuperação completa ou parcial
- **Auditoria:** Log de todas as ações do sistema
- **Métricas:** Monitoramento de performance
- **Health Checks:** Verificação de saúde do sistema

## 📊 Configuração Avançada

### Estações com Metadados Completos
```yaml
stations:
  Two Mouths:
    type: primary
    capacity: 10000
    location:
      latitude: -23.5505
      longitude: -46.6333
    sensors:
      pH:
        model: "PHE-45P"
        manufacturer: "Hach"
        calibration_interval: 7
        accuracy: "±0.1 pH"
    maintenance:
      next_inspection: "2024-02-15"
      status: "operational"
```

### Limites Operacionais Avançados
```yaml
thresholds:
  pH:
    min: 6.5
    max: 8.5
    critical_min: 6.0
    critical_max: 9.0
    target: 7.0
    tolerance: 0.5
    regulatory_limit: true
    enforcement_level: "mandatory"
```

### Sistema de Alertas Configurável
```yaml
alerts:
  escalation:
    - level: "critical"
      delay: 0
      notify: ["email", "slack", "sms", "teams"]
      auto_resolve: false
      escalation_after: 300
```

## 🔍 Monitoramento e Métricas

### Métricas do Sistema
- **CPU:** Utilização e contagem de cores
- **Memória:** Uso, disponível, total
- **Disco:** Espaço livre, total, percentual
- **Rede:** Bytes enviados/recebidos, pacotes

### Health Checks
- **Database Connection:** Verificação de conectividade
- **Data Freshness:** Idade dos dados
- **Alert System:** Status do sistema de alertas
- **External APIs:** Conectividade com APIs externas

### Logs Estruturados
- **Níveis:** DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Formato:** JSON estruturado
- **Rotação:** Automática por data/tamanho
- **Compressão:** Gzip para otimização

## 🧪 Testes e Validação

### Script de Teste Completo
```bash
python test_advanced_features.py
```

### Testes Incluídos
- ✅ DataLoader com cache e metadados
- ✅ DataPreprocessor com features avançadas
- ✅ AnomalyDetector com múltiplos métodos
- ✅ Sistema de backup e auditoria
- ✅ Sistema de alertas avançado

## 📈 Performance e Otimização

### Cache Inteligente
- **LRU Cache:** Para metadados e consultas frequentes
- **Memory Cache:** Para dados em memória
- **TTL:** Time-to-live configurável
- **Invalidation:** Invalidação automática

### Otimizações
- **Paralelização:** Processamento paralelo de dados
- **Compressão:** Dados comprimidos em memória
- **Lazy Loading:** Carregamento sob demanda
- **Connection Pooling:** Pool de conexões de banco

## 🔒 Segurança

### Autenticação e Autorização
- **JWT Tokens:** Autenticação stateless
- **Role-based Access:** Controle de acesso por função
- **Session Management:** Gerenciamento de sessões
- **Password Policies:** Políticas de senha configuráveis

### Criptografia
- **Data at Rest:** Criptografia de dados armazenados
- **Data in Transit:** HTTPS/TLS para comunicação
- **Backup Encryption:** Criptografia de backups
- **Key Management:** Gerenciamento seguro de chaves

## 🚀 Deploy e Produção

### Configuração de Produção
```yaml
system:
  enable_authentication: true
  session_timeout: 3600
  max_login_attempts: 5
  backup_retention_days: 30
  encryption_enabled: true
```

### Variáveis de Ambiente
```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export API_SERVER_PORT=8000
export DATABASE_URL=sqlite:///water_treatment.db
export ENCRYPTION_KEY=your-encryption-key
```

### Docker Support
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py"]
```

## 📚 Documentação da API

### Acessar Documentação
- **Swagger UI:** `http://localhost:8000/api/v1/docs`
- **OpenAPI Spec:** `http://localhost:8000/api/v1/openapi.json`

### Exemplos de Uso
```python
import requests

# Health check
response = requests.get('http://localhost:8000/api/v1/health')
print(response.json())

# Obter dados
response = requests.get('http://localhost:8000/api/v1/data?station=Two Mouths&limit=100')
data = response.json()

# Detectar anomalias
payload = {
    'station': 'Two Mouths',
    'method': 'isolation_forest',
    'contamination': 0.1
}
response = requests.post('http://localhost:8000/api/v1/analysis/anomalies', json=payload)
anomalies = response.json()
```

## 🎯 Roadmap Futuro

### Próximas Funcionalidades
- [ ] **Machine Learning Pipeline:** Pipeline automatizado de ML
- [ ] **Real-time Streaming:** Integração com Apache Kafka
- [ ] **Microservices:** Arquitetura de microsserviços
- [ ] **Kubernetes:** Deploy em containers
- [ ] **Grafana Integration:** Dashboards de monitoramento
- [ ] **Mobile App:** Aplicativo móvel nativo
- [ ] **IoT Integration:** Conectividade com sensores IoT
- [ ] **Blockchain:** Auditoria imutável de dados

### Melhorias Planejadas
- [ ] **Performance:** Otimização de consultas
- [ ] **Scalability:** Suporte a milhões de registros
- [ ] **UI/UX:** Interface mais intuitiva
- [ ] **Testing:** Cobertura de testes 100%
- [ ] **Documentation:** Documentação completa
- [ ] **Internationalization:** Suporte a múltiplos idiomas

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma feature branch
3. Implemente suas mudanças
4. Adicione testes
5. Documente as mudanças
6. Abra um Pull Request

### Padrões de Código
- **PEP 8:** Seguir padrões Python
- **Type Hints:** Usar anotações de tipo
- **Docstrings:** Documentar todas as funções
- **Tests:** Cobertura de testes > 90%
- **Linting:** Black, flake8, mypy

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

### Documentação
- **README:** Este arquivo
- **API Docs:** `/api/v1/docs`
- **Code:** Comentários e docstrings
- **Config:** Arquivos YAML

### Contato
- **Issues:** GitHub Issues
- **Email:** [seu-email@exemplo.com]
- **Documentation:** [link-para-docs]

---

**Desenvolvido com ❤️ usando Streamlit, Python e tecnologias de análise de dados avançadas**

*Sistema de Monitoramento Avançado v3.0 - 2024*
