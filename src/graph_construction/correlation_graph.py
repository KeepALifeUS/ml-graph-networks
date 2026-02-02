"""
Correlation-based Graph Construction for Crypto Markets
========================================================

Enterprise-grade graph construction algorithms based on price correlations,
volume relationships, and market dynamics for cryptocurrency trading analysis.

Features:
- Pearson, Spearman, and Kendall correlation graphs
- Dynamic correlation tracking over time windows
- Market regime-aware correlation graphs
- Volatility-adjusted correlations
- Cross-asset correlation networks
- Production-ready scalability with enterprise patterns

Author: ML-Framework ML Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationGraphConfig:
    """
    Конфигурация для построения correlation-based графов
    
    Comprehensive Configuration Management
    """
    # Correlation parameters
    correlation_method: str = 'pearson'  # pearson, spearman, kendall, distance_correlation
    time_window: int = 30  # Временное окно для корреляции (дни)
    min_correlation: float = 0.3  # Минимальная корреляция для создания ребра
    
    # Dynamic correlation parameters
    use_rolling_correlation: bool = True
    rolling_window: int = 20  # Окно для rolling correlation
    correlation_decay: float = 0.95  # Экспоненциальное затухание для динамических корреляций
    
    # Market regime awareness
    use_market_regimes: bool = True
    volatility_threshold: float = 0.02  # Пороговое значение волатильности
    regime_window: int = 10  # Окно для определения режима рынка
    
    # Graph construction parameters
    max_edges_per_node: int = 10  # Максимальное количество рёбер на узел
    edge_weight_transform: str = 'absolute'  # absolute, squared, tanh
    use_edge_attributes: bool = True
    
    # Network filtering
    apply_threshold_filtering: bool = True
    apply_topological_filtering: bool = False
    minimum_spanning_tree: bool = False  # Создавать MST вместо полного графа
    
    # Advanced features
    adjust_for_volatility: bool = True
    use_partial_correlations: bool = False  # Частичные корреляции
    include_lag_correlations: bool = False  # Лаговые корреляции
    max_lag: int = 5  # Максимальный лаг для лаговых корреляций

class CorrelationCalculator:
    """
    Вычислитель различных типов корреляций для crypto assets
    
    Strategy Pattern для correlation methods
    """
    
    def __init__(self, method: str = 'pearson'):
        self.method = method
        self.correlation_functions = {
            'pearson': self._pearson_correlation,
            'spearman': self._spearman_correlation,
            'kendall': self._kendall_correlation,
            'distance_correlation': self._distance_correlation,
            'mutual_information': self._mutual_information_correlation
        }
    
    def compute_correlation_matrix(
        self, 
        data: pd.DataFrame, 
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Вычисление корреляционной матрицы
        
        Args:
            data: DataFrame с ценовыми данными [time, assets]
            method: Метод корреляции (если None - используется self.method)
            
        Returns:
            np.ndarray: Корреляционная матрица [n_assets, n_assets]
        """
        method = method or self.method
        
        if method not in self.correlation_functions:
            raise ValueError(f"Неизвестный метод корреляции: {method}")
        
        return self.correlation_functions[method](data)
    
    def _pearson_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Pearson корреляция"""
        # Убираем NaN значения
        data_clean = data.dropna()
        if data_clean.empty:
            logger.warning("Нет данных после удаления NaN")
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='pearson').values
    
    def _spearman_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Spearman ранговая корреляция"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='spearman').values
    
    def _kendall_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Kendall tau корреляция"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='kendall').values
    
    def _distance_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Distance correlation (нелинейная зависимость)"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        n_assets = data_clean.shape[1]
        dcorr_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Упрощённая версия distance correlation
                x = data_clean.iloc[:, i].values
                y = data_clean.iloc[:, j].values
                
                # Центрированные данные
                x_centered = x - np.mean(x)
                y_centered = y - np.mean(y)
                
                # Distance correlation approximation
                dcorr = np.corrcoef(np.abs(x_centered), np.abs(y_centered))[0, 1]
                dcorr = dcorr if not np.isnan(dcorr) else 0.0
                
                dcorr_matrix[i, j] = dcorr
                dcorr_matrix[j, i] = dcorr
        
        return dcorr_matrix
    
    def _mutual_information_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Mutual information based correlation"""
        from sklearn.feature_selection import mutual_info_regression
        
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        n_assets = data_clean.shape[1]
        mi_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                try:
                    x = data_clean.iloc[:, i].values.reshape(-1, 1)
                    y = data_clean.iloc[:, j].values
                    
                    mi = mutual_info_regression(x, y)[0]
                    mi_normalized = 2 * mi / (np.var(data_clean.iloc[:, i]) + np.var(data_clean.iloc[:, j]))
                    mi_normalized = min(mi_normalized, 1.0)  # Ограничиваем [0, 1]
                    
                    mi_matrix[i, j] = mi_normalized
                    mi_matrix[j, i] = mi_normalized
                    
                except Exception as e:
                    logger.warning(f"Ошибка в mutual information для {i}, {j}: {e}")
                    mi_matrix[i, j] = 0.0
                    mi_matrix[j, i] = 0.0
        
        return mi_matrix
    
    def compute_rolling_correlations(
        self, 
        data: pd.DataFrame, 
        window: int,
        method: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Rolling корреляции по временным окнам
        
        Returns:
            List[np.ndarray]: Список корреляционных матриц для каждого временного шага
        """
        method = method or self.method
        rolling_correlations = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            corr_matrix = self.compute_correlation_matrix(window_data, method)
            rolling_correlations.append(corr_matrix)
        
        return rolling_correlations
    
    def compute_lagged_correlations(
        self, 
        data: pd.DataFrame, 
        max_lag: int = 5,
        method: Optional[str] = None
    ) -> Dict[int, np.ndarray]:
        """
        Лаговые корреляции между активами
        
        Returns:
            Dict[int, np.ndarray]: {lag: correlation_matrix}
        """
        method = method or self.method
        lagged_correlations = {}
        
        for lag in range(1, max_lag + 1):
            lagged_data = data.copy()
            
            # Создаём лаговые версии данных
            for col in data.columns:
                lagged_data[f"{col}_lag{lag}"] = data[col].shift(lag)
            
            # Вычисляем корреляции между исходными и лаговыми данными
            original_cols = data.columns
            lagged_cols = [f"{col}_lag{lag}" for col in original_cols]
            
            cross_corr_data = pd.concat([
                lagged_data[original_cols], 
                lagged_data[lagged_cols]
            ], axis=1).dropna()
            
            if not cross_corr_data.empty:
                full_corr = self.compute_correlation_matrix(cross_corr_data, method)
                # Извлекаем кросс-корреляции (исходные vs лаговые)
                n_assets = len(original_cols)
                cross_correlations = full_corr[:n_assets, n_assets:]
                lagged_correlations[lag] = cross_correlations
            else:
                lagged_correlations[lag] = np.zeros((len(original_cols), len(original_cols)))
        
        return lagged_correlations

class MarketRegimeDetector:
    """
    Детектор рыночных режимов для адаптивных корреляций
    
    Market Intelligence Module
    """
    
    def __init__(self, volatility_threshold: float = 0.02, window: int = 10):
        self.volatility_threshold = volatility_threshold
        self.window = window
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Определение рыночного режима (низкая/высокая волатильность)
        
        Args:
            data: DataFrame с ценовыми данными
            
        Returns:
            pd.Series: Режимы рынка ('low_vol', 'high_vol')
        """
        # Вычисляем волатильность (rolling std доходностей)
        returns = data.pct_change().dropna()
        volatility = returns.rolling(window=self.window).std().mean(axis=1)
        
        # Классификация режимов
        regimes = pd.Series(index=volatility.index, dtype=str)
        regimes[volatility <= self.volatility_threshold] = 'low_vol'
        regimes[volatility > self.volatility_threshold] = 'high_vol'
        
        return regimes
    
    def get_regime_correlations(
        self, 
        data: pd.DataFrame, 
        calculator: CorrelationCalculator
    ) -> Dict[str, np.ndarray]:
        """
        Корреляции в разных рыночных режимах
        
        Returns:
            Dict[str, np.ndarray]: {'low_vol': corr_matrix, 'high_vol': corr_matrix}
        """
        regimes = self.detect_regime(data)
        regime_correlations = {}
        
        for regime in ['low_vol', 'high_vol']:
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 10:  # Достаточно данных
                regime_correlations[regime] = calculator.compute_correlation_matrix(regime_data)
            else:
                logger.warning(f"Недостаточно данных для режима {regime}")
                regime_correlations[regime] = np.eye(data.shape[1])
        
        return regime_correlations

class CorrelationGraphBuilder:
    """
    Основной класс для построения correlation-based графов
    
    Builder Pattern для graph construction
    """
    
    def __init__(self, config: CorrelationGraphConfig):
        self.config = config
        self.calculator = CorrelationCalculator(config.correlation_method)
        
        if config.use_market_regimes:
            self.regime_detector = MarketRegimeDetector(
                config.volatility_threshold, 
                config.regime_window
            )
        
        logger.info(f"Инициализирован CorrelationGraphBuilder с методом {config.correlation_method}")
    
    def build_correlation_graph(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        node_features: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> Data:
        """
        Построение корреляционного графа из ценовых данных
        
        Args:
            price_data: DataFrame с ценами активов [time, assets]
            volume_data: Опциональные данные по объёмам
            node_features: Дополнительные признаки узлов
            asset_names: Названия активов
            
        Returns:
            Data: PyTorch Geometric Data объект
        """
        if asset_names is None:
            asset_names = price_data.columns.tolist()
        
        # Вычисление корреляционной матрицы
        correlation_matrix = self._compute_adaptive_correlations(price_data, volume_data)
        
        # Построение графа из корреляционной матрицы
        edge_index, edge_weights, edge_attr = self._matrix_to_graph(
            correlation_matrix, 
            asset_names
        )
        
        # Node features
        if node_features is None:
            node_features = self._extract_node_features(price_data, volume_data)
        
        # Создание PyG Data объекта
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weights,
            edge_attr=edge_attr if self.config.use_edge_attributes else None
        )
        
        # Добавление метаданных
        data.asset_names = asset_names
        data.correlation_matrix = correlation_matrix
        data.num_nodes = len(asset_names)
        
        return data
    
    def _compute_adaptive_correlations(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Вычисление адаптивных корреляций с учётом конфигурации"""
        
        if self.config.use_rolling_correlation:
            # Rolling correlations
            rolling_corrs = self.calculator.compute_rolling_correlations(
                price_data, 
                self.config.rolling_window,
                self.config.correlation_method
            )
            
            if rolling_corrs:
                # Экспоненциально взвешенное среднее корреляций
                weights = np.array([
                    self.config.correlation_decay ** (len(rolling_corrs) - i - 1)
                    for i in range(len(rolling_corrs))
                ])
                weights = weights / weights.sum()
                
                correlation_matrix = np.average(rolling_corrs, axis=0, weights=weights)
            else:
                correlation_matrix = self.calculator.compute_correlation_matrix(price_data)
        
        elif self.config.use_market_regimes:
            # Режим-зависимые корреляции
            regime_correlations = self.regime_detector.get_regime_correlations(
                price_data, self.calculator
            )
            
            # Определяем текущий режим по последним данным
            current_regime = self.regime_detector.detect_regime(
                price_data.tail(self.config.regime_window)
            ).mode().iloc[0]
            
            correlation_matrix = regime_correlations.get(current_regime, 
                                                       self.calculator.compute_correlation_matrix(price_data))
        
        else:
            # Стандартная корреляция
            correlation_matrix = self.calculator.compute_correlation_matrix(price_data)
        
        # Adjustment for volatility
        if self.config.adjust_for_volatility and volume_data is not None:
            correlation_matrix = self._adjust_for_volatility(
                correlation_matrix, price_data, volume_data
            )
        
        # Partial correlations
        if self.config.use_partial_correlations:
            correlation_matrix = self._compute_partial_correlations(
                price_data, correlation_matrix
            )
        
        return correlation_matrix
    
    def _adjust_for_volatility(
        self, 
        correlation_matrix: np.ndarray,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> np.ndarray:
        """Корректировка корреляций на волатильность"""
        returns = price_data.pct_change().dropna()
        volatilities = returns.std().values
        
        # Weighted correlation by inverse volatility
        vol_weights = 1.0 / (volatilities + 1e-8)
        vol_weights = vol_weights / vol_weights.sum()
        
        # Apply volatility adjustment
        adjusted_corr = correlation_matrix.copy()
        for i in range(len(vol_weights)):
            for j in range(len(vol_weights)):
                if i != j:
                    vol_factor = np.sqrt(vol_weights[i] * vol_weights[j])
                    adjusted_corr[i, j] *= vol_factor
        
        return adjusted_corr
    
    def _compute_partial_correlations(
        self, 
        price_data: pd.DataFrame,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Частичные корреляции (удаляем влияние других переменных)"""
        try:
            # Precision matrix (inverse of covariance)
            returns = price_data.pct_change().dropna()
            cov_matrix = returns.cov().values
            
            # Regularization для стабильности
            reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
            precision_matrix = np.linalg.inv(reg_cov)
            
            # Partial correlations from precision matrix
            partial_corr = np.zeros_like(precision_matrix)
            for i in range(len(precision_matrix)):
                for j in range(len(precision_matrix)):
                    if i != j:
                        partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
                    else:
                        partial_corr[i, j] = 1.0
            
            return partial_corr
            
        except np.linalg.LinAlgError:
            logger.warning("Не удалось вычислить частичные корреляции, используем обычные")
            return correlation_matrix
    
    def _matrix_to_graph(
        self, 
        correlation_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Преобразование корреляционной матрицы в граф"""
        
        n_assets = len(asset_names)
        edges = []
        weights = []
        edge_attributes = []
        
        # Создание рёбер на основе корреляций
        for i in range(n_assets):
            # Получаем корреляции для узла i
            node_correlations = []
            for j in range(n_assets):
                if i != j:
                    corr_value = correlation_matrix[i, j]
                    # Применяем трансформацию веса
                    weight = self._transform_edge_weight(corr_value)
                    
                    if abs(weight) >= self.config.min_correlation:
                        node_correlations.append((j, weight, corr_value))
            
            # Сортируем по убыванию весов и берём топ-K
            node_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top_connections = node_correlations[:self.config.max_edges_per_node]
            
            for j, weight, raw_corr in top_connections:
                edges.append([i, j])
                weights.append(weight)
                
                if self.config.use_edge_attributes:
                    # Дополнительные атрибуты рёбер
                    edge_attr = [
                        raw_corr,  # Исходная корреляция
                        abs(raw_corr),  # Абсолютная корреляция
                        1.0 if raw_corr > 0 else -1.0,  # Знак корреляции
                        weight  # Трансформированный вес
                    ]
                    edge_attributes.append(edge_attr)
        
        # Minimum Spanning Tree если требуется
        if self.config.minimum_spanning_tree:
            edges, weights, edge_attributes = self._build_mst(
                correlation_matrix, edges, weights, edge_attributes
            )
        
        # Конвертация в PyTorch tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(weights, dtype=torch.float32)
            
            if self.config.use_edge_attributes and edge_attributes:
                edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
            else:
                edge_attr = None
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
            edge_attr = None
        
        return edge_index, edge_weights, edge_attr
    
    def _transform_edge_weight(self, correlation: float) -> float:
        """Трансформация веса ребра"""
        if self.config.edge_weight_transform == 'absolute':
            return abs(correlation)
        elif self.config.edge_weight_transform == 'squared':
            return correlation ** 2
        elif self.config.edge_weight_transform == 'tanh':
            return np.tanh(abs(correlation))
        else:
            return correlation
    
    def _build_mst(
        self, 
        correlation_matrix: np.ndarray,
        edges: List[List[int]], 
        weights: List[float],
        edge_attributes: List[List[float]]
    ) -> Tuple[List[List[int]], List[float], List[List[float]]]:
        """Построение Minimum Spanning Tree"""
        # Создаём NetworkX граф
        G = nx.Graph()
        
        for i, (edge, weight) in enumerate(zip(edges, weights)):
            G.add_edge(edge[0], edge[1], weight=1.0 - abs(weight))  # Инвертируем для MST
        
        # Строим MST
        mst = nx.minimum_spanning_tree(G)
        
        # Извлекаем рёбра MST
        mst_edges = []
        mst_weights = []
        mst_attributes = []
        
        for i, (u, v) in enumerate(mst.edges()):
            # Находим исходный вес
            original_weight = abs(correlation_matrix[u, v])
            mst_edges.append([u, v])
            mst_weights.append(original_weight)
            
            if self.config.use_edge_attributes:
                raw_corr = correlation_matrix[u, v]
                edge_attr = [
                    raw_corr,
                    abs(raw_corr),
                    1.0 if raw_corr > 0 else -1.0,
                    original_weight
                ]
                mst_attributes.append(edge_attr)
        
        return mst_edges, mst_weights, mst_attributes
    
    def _extract_node_features(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Извлечение признаков узлов из данных"""
        features = []
        
        for asset in price_data.columns:
            asset_prices = price_data[asset].dropna()
            asset_returns = asset_prices.pct_change().dropna()
            
            asset_features = [
                # Статистические характеристики
                asset_returns.mean(),  # Средняя доходность
                asset_returns.std(),   # Волатильность
                asset_returns.skew(),  # Асимметрия
                asset_returns.kurtosis(),  # Эксцесс
                
                # Риск-метрики
                asset_returns.quantile(0.05),  # VaR 5%
                asset_returns.quantile(0.95),  # VaR 95%
                
                # Технические индикаторы
                asset_prices.iloc[-1] / asset_prices.iloc[0] - 1,  # Общая доходность
                len(asset_returns[asset_returns > 0]) / len(asset_returns),  # % положительных дней
            ]
            
            # Добавляем объёмные характеристики если доступны
            if volume_data is not None and asset in volume_data.columns:
                asset_volumes = volume_data[asset].dropna()
                asset_features.extend([
                    asset_volumes.mean(),  # Средний объём
                    asset_volumes.std(),   # Стандартное отклонение объёма
                    asset_volumes.iloc[-1] / asset_volumes.mean()  # Относительный текущий объём
                ])
            else:
                # Заполняем нулями если нет данных по объёмам
                asset_features.extend([0.0, 0.0, 1.0])
            
            features.append(asset_features)
        
        # Нормализация признаков
        features_array = np.array(features)
        
        # Заменяем NaN и inf значения
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Стандартизация
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        return features_normalized.astype(np.float32)
    
    def build_dynamic_correlation_graph(
        self, 
        price_data: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        node_features: Optional[np.ndarray] = None
    ) -> List[Data]:
        """
        Построение динамических корреляционных графов
        
        Returns:
            List[Data]: Список графов для каждого временного шага
        """
        dynamic_graphs = []
        
        for i, timestamp in enumerate(timestamps):
            # Определяем временное окно
            start_idx = max(0, i - self.config.time_window)
            end_idx = i + 1
            
            window_data = price_data.iloc[start_idx:end_idx]
            
            if len(window_data) < 5:  # Минимальное количество наблюдений
                continue
            
            # Строим граф для этого окна
            graph = self.build_correlation_graph(
                window_data,
                node_features=node_features[i:i+1] if node_features is not None else None
            )
            
            # Добавляем временную метку
            graph.timestamp = timestamp
            graph.window_start = window_data.index[0]
            graph.window_end = window_data.index[-1]
            
            dynamic_graphs.append(graph)
        
        logger.info(f"Построено {len(dynamic_graphs)} динамических графов")
        return dynamic_graphs

def create_correlation_graph(
    price_data: pd.DataFrame,
    volume_data: Optional[pd.DataFrame] = None,
    correlation_method: str = 'pearson',
    min_correlation: float = 0.3,
    **kwargs
) -> Data:
    """
    Factory функция для быстрого создания корреляционного графа
    
    Simple Factory для graph creation
    """
    config = CorrelationGraphConfig(
        correlation_method=correlation_method,
        min_correlation=min_correlation,
        **kwargs
    )
    
    builder = CorrelationGraphBuilder(config)
    return builder.build_correlation_graph(price_data, volume_data)

# Экспорт основных классов
__all__ = [
    'CorrelationGraphConfig',
    'CorrelationCalculator',
    'MarketRegimeDetector',
    'CorrelationGraphBuilder',
    'create_correlation_graph'
]