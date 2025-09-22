"""
Módulo de Localização / Localization Module
Gerencia traduções entre português e inglês
"""

import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, Any

class Localization:
    """Classe para gerenciar traduções"""

    def __init__(self):
        """Inicializar sistema de localização"""
        self.config_path = Path(__file__).parent.parent / "config" / "localization.yaml"
        self.translations = self.load_translations()

        # Definir idioma padrão como inglês
        if 'language' not in st.session_state:
            st.session_state.language = 'en'

    def load_translations(self) -> Dict[str, Any]:
        """Carregar traduções do arquivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            st.error(f"Erro ao carregar traduções: {e}")
            return {'pt': {}, 'en': {}}

    def get_language(self) -> str:
        """Obter idioma atual"""
        return 'en'  # Força inglês

    def set_language(self, language: str):
        """Definir idioma"""
        if language in ['pt', 'en']:
            st.session_state.language = language

    def t(self, key: str, **kwargs) -> str:
        """
        Traduzir texto baseado na chave

        Args:
            key: Chave da tradução
            **kwargs: Parâmetros para formatação de string

        Returns:
            Texto traduzido
        """
        language = self.get_language()

        try:
            text = self.translations[language].get(key, key)
            if kwargs:
                return text.format(**kwargs)
            return text
        except (KeyError, AttributeError):
            return key

    def render_language_selector(self):
        """Renderizar seletor de idioma"""
        languages = {
            'pt': '🇧🇷 Português',
            'en': '🇺🇸 English'
        }

        current_lang = self.get_language()

        selected_lang = st.selectbox(
            self.t('language'),
            options=['pt', 'en'],
            index=0 if current_lang == 'pt' else 1,
            format_func=lambda x: languages[x],
            key='language_selector'
        )

        # Verificar se houve mudança e atualizar
        if 'language_selector' in st.session_state and st.session_state.language_selector != current_lang:
            self.set_language(st.session_state.language_selector)
            st.rerun()

    def get_page_titles(self) -> Dict[str, str]:
        """Obter títulos das páginas traduzidos"""
        return {
            'overview': self.t('overview'),
            'time_series': self.t('time_series'),
            'predictions': self.t('predictions'),
            'anomalies': self.t('anomalies'),
            'reports': self.t('reports'),
            'settings': self.t('settings')
        }

# Instância global de localização
localization = Localization()

# Função de conveniência para tradução
def t(key: str, **kwargs) -> str:
    """Função de conveniência para tradução"""
    return localization.t(key, **kwargs)

# Função para obter idioma atual
def get_language() -> str:
    """Função de conveniência para obter idioma"""
    return localization.get_language()

# Função para definir idioma
def set_language(language: str):
    """Função de conveniência para definir idioma"""
    localization.set_language(language)