# 🚀 Guia de Instalação - Sistema de Monitoramento

## 📋 Pré-requisitos

### Sistema Operacional
- **Windows**: 10 ou superior
- **macOS**: 10.14 ou superior  
- **Linux**: Ubuntu 18.04+ ou equivalente

### Python
- **Versão**: Python 3.8 ou superior
- **Verificar**: `python --version` ou `python3 --version`

### Memória e Armazenamento
- **RAM**: Mínimo 4GB (recomendado 8GB+)
- **Armazenamento**: 2GB livres
- **CPU**: Processador moderno (recomendado)

## 🔧 Instalação Passo a Passo

### 1. Clone o Repositório

```bash
# Via HTTPS
git clone https://github.com/seu-usuario/StreamlitApp.git

# Via SSH (se configurado)
git clone git@github.com:seu-usuario/StreamlitApp.git

# Navegar para o diretório
cd StreamlitApp
```

### 2. Criar Ambiente Virtual (Recomendado)

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Ou instalar manualmente as principais:
pip install streamlit pandas numpy plotly scipy statsmodels scikit-learn
```

### 4. Verificar Instalação

```bash
# Testar Streamlit
streamlit --version

# Testar módulos Python
python -c "import pandas, numpy, plotly; print('✅ Dependências OK')"
```

### 5. Gerar Dados Sintéticos (Opcional)

```bash
# Gerar dados de demonstração
python generate_synthetic_data.py
```

### 6. Executar a Aplicação

```bash
# Modo desenvolvimento
streamlit run app.py

# Modo produção
streamlit run app.py --server.headless true --server.port 8501
```

## 🌐 Acesso à Aplicação

Após executar o comando acima, acesse:

- **URL Local**: http://localhost:8501
- **URL Rede**: http://seu-ip:8501

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# Windows (PowerShell)
$env:STREAMLIT_SERVER_PORT="8501"
$env:STREAMLIT_SERVER_ADDRESS="0.0.0.0"

# macOS/Linux
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Arquivo de Configuração Streamlit

Crie um arquivo `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🐳 Instalação com Docker (Opcional)

### 1. Criar Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "0.0.0.0"]
```

### 2. Build e Executar

```bash
# Build da imagem
docker build -t water-monitoring .

# Executar container
docker run -p 8501:8501 water-monitoring
```

## 🔍 Solução de Problemas

### Erro: "ModuleNotFoundError"

```bash
# Verificar se o ambiente virtual está ativo
which python  # macOS/Linux
where python  # Windows

# Reinstalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### Erro: "Port already in use"

```bash
# Usar porta diferente
streamlit run app.py --server.port 8502

# Ou matar processo na porta
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

### Erro: "Permission denied"

```bash
# Dar permissões de execução
chmod +x app.py

# Ou executar com sudo (não recomendado)
sudo streamlit run app.py
```

### Erro: "Memory error"

```bash
# Reduzir tamanho dos dados
# Editar generate_synthetic_data.py
# Reduzir days=90 para days=30
```

## 📊 Verificação da Instalação

### Teste Básico

```bash
# Executar exemplo
python example_usage.py
```

### Teste Completo

1. Acesse http://localhost:8501
2. Navegue pelas páginas do dashboard
3. Teste as funcionalidades:
   - Carregamento de dados
   - Visualizações
   - Detecção de anomalias
   - Geração de relatórios

## 🔧 Dependências Opcionais

### Para Funcionalidades Avançadas

```bash
# Prophet (previsões)
pip install prophet

# PyOD (detecção de anomalias)
pip install pyod

# XGBoost (machine learning)
pip install xgboost

# OpenPyXL (exportação Excel)
pip install openpyxl
```

### Para Desenvolvimento

```bash
# Jupyter (notebooks)
pip install jupyter

# Black (formatação de código)
pip install black

# Pytest (testes)
pip install pytest
```

## 🚀 Deploy em Produção

### 1. Configurar Servidor

```bash
# Instalar nginx (Ubuntu/Debian)
sudo apt update
sudo apt install nginx

# Configurar proxy reverso
sudo nano /etc/nginx/sites-available/streamlit
```

### 2. Configuração Nginx

```nginx
server {
    listen 80;
    server_name seu-dominio.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Serviço Systemd

```bash
# Criar arquivo de serviço
sudo nano /etc/systemd/system/streamlit.service
```

```ini
[Unit]
Description=Streamlit Water Monitoring App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/StreamlitApp
Environment=PATH=/path/to/StreamlitApp/venv/bin
ExecStart=/path/to/StreamlitApp/venv/bin/streamlit run app.py --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
```

### 4. Ativar Serviço

```bash
# Recarregar systemd
sudo systemctl daemon-reload

# Habilitar serviço
sudo systemctl enable streamlit

# Iniciar serviço
sudo systemctl start streamlit

# Verificar status
sudo systemctl status streamlit
```

## 📞 Suporte

### Problemas Comuns

1. **Erro de importação**: Verificar se todas as dependências estão instaladas
2. **Porta ocupada**: Usar porta diferente ou matar processo
3. **Dados não carregam**: Verificar se os arquivos CSV existem
4. **Interface não carrega**: Verificar se o Streamlit está rodando

### Logs e Debug

```bash
# Executar com debug
streamlit run app.py --logger.level debug

# Ver logs do sistema
# Windows: Event Viewer
# macOS: Console.app
# Linux: journalctl -u streamlit
```

### Contato

- **Issues**: GitHub Issues
- **Email**: suporte@exemplo.com
- **Documentação**: README.md

---

**✅ Instalação concluída com sucesso!**

Agora você pode acessar o sistema em http://localhost:8501
