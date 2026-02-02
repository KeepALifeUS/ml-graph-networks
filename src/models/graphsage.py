"""
GraphSAGE Implementation for Crypto Trading
============================================

Production-ready GraphSAGE with advanced sampling strategies 
optimized for large-scale crypto market graphs with enterprise patterns.

Features:
- Inductive learning capabilities
- Multiple sampling strategies (uniform, random walk, importance sampling)
- Hierarchical neighborhood aggregation
- Scalable batch processing
- Production-ready optimization
- Crypto-specific aggregation functions

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data, Batch, NeighborSampler
from torch_geometric.loader import NeighborLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
import random
from collections import defaultdict
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class GraphSAGEConfig:
    """
    Конфигурация для GraphSAGE модели
    
    Configuration Management
    """
    input_dim: int = 64
    hidden_dims: List[int] = None  
    output_dim: int = 1
    num_layers: int = 3
    dropout_rate: float = 0.2
    activation: str = 'relu'
    aggregation: str = 'mean'  # mean, max, lstm, pool
    use_batch_norm: bool = True
    use_residual: bool = True
    normalize: bool = True  # L2 normalization after each layer
    
    # Sampling parameters
    neighbor_sizes: List[int] = None  # Размеры neighborhood для каждого слоя
    sampling_strategy: str = 'uniform'  # uniform, random_walk, importance
    walk_length: int = 3  # Для random walk sampling
    num_walks: int = 10  # Количество walks для каждого узла
    
    # Training parameters  
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    
    # Advanced features
    use_edge_weights: bool = True
    use_attention: bool = False  # Attention-based aggregation
    temperature: float = 1.0  # Temperature для softmax в attention
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [
                max(32, self.input_dim // (2**i)) 
                for i in range(self.num_layers - 1)
            ]
        
        if self.neighbor_sizes is None:
            # Уменьшающиеся размеры neighborhood по слоям
            self.neighbor_sizes = [25, 15, 10][:self.num_layers]

class AdvancedSampler:
    """
    Продвинутый sampler для GraphSAGE с различными стратегиями
    
    Strategy Pattern для sampling
    """
    
    def __init__(self, config: GraphSAGEConfig):
        self.config = config
        self.sampling_functions = {
            'uniform': self._uniform_sampling,
            'random_walk': self._random_walk_sampling, 
            'importance': self._importance_sampling,
            'crypto_correlation': self._crypto_correlation_sampling
        }
    
    def sample_neighbors(
        self, 
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        layer_idx: int,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выборка соседей для указанных узлов
        
        Args:
            edge_index: Индексы рёбер графа
            node_indices: Индексы узлов для которых нужны соседи
            layer_idx: Индекс слоя (влияет на размер neighborhood)
            node_features: Признаки узлов (для importance sampling)
            edge_weights: Веса рёбер
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (sampled_edges, sampled_weights)
        """
        strategy = self.config.sampling_strategy
        sampling_fn = self.sampling_functions.get(strategy, self._uniform_sampling)
        
        return sampling_fn(
            edge_index, node_indices, layer_idx, 
            node_features, edge_weights
        )
    
    def _uniform_sampling(
        self, 
        edge_index: torch.Tensor,
        node_indices: torch.Tensor, 
        layer_idx: int,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform sampling соседей"""
        max_neighbors = self.config.neighbor_sizes[min(layer_idx, len(self.config.neighbor_sizes) - 1)]
        
        # Создание adjacency list
        adj_list = defaultdict(list)
        edge_weight_dict = {}
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
            if edge_weights is not None:
                edge_weight_dict[(src, dst)] = edge_weights[i]
        
        sampled_edges = []
        sampled_weights = []
        
        for node in node_indices:
            node = node.item()
            neighbors = adj_list[node]
            
            if len(neighbors) > max_neighbors:
                # Uniform sampling
                sampled_neighbors = random.sample(neighbors, max_neighbors)
            else:
                sampled_neighbors = neighbors
            
            # Добавление рёбер
            for neighbor in sampled_neighbors:
                sampled_edges.append([node, neighbor])
                if edge_weights is not None:
                    weight = edge_weight_dict.get((node, neighbor), 1.0)
                    sampled_weights.append(weight)
        
        if sampled_edges:
            sampled_edges = torch.tensor(sampled_edges, dtype=torch.long).t()
            if edge_weights is not None and sampled_weights:
                sampled_weights = torch.tensor(sampled_weights, dtype=torch.float)
            else:
                sampled_weights = None
        else:
            sampled_edges = torch.empty((2, 0), dtype=torch.long)
            sampled_weights = None
        
        return sampled_edges, sampled_weights
    
    def _random_walk_sampling(
        self,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        layer_idx: int,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random walk sampling для лучшего сохранения структуры"""
        max_neighbors = self.config.neighbor_sizes[min(layer_idx, len(self.config.neighbor_sizes) - 1)]
        
        # Adjacency list с весами
        adj_list = defaultdict(list)
        edge_weight_dict = {}
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weights[i].item() if edge_weights is not None else 1.0
            adj_list[src].append((dst, weight))
            edge_weight_dict[(src, dst)] = weight
        
        sampled_edges = []
        sampled_weights = []
        
        for start_node in node_indices:
            start_node = start_node.item()
            neighbors_set = set()
            
            # Выполнение нескольких random walks
            for _ in range(self.config.num_walks):
                current_node = start_node
                
                # Random walk длиной walk_length
                for step in range(self.config.walk_length):
                    if current_node in adj_list:
                        neighbors_with_weights = adj_list[current_node]
                        if neighbors_with_weights:
                            # Weighted sampling
                            neighbors, weights = zip(*neighbors_with_weights)
                            weights = np.array(weights)
                            probs = weights / weights.sum()
                            
                            next_node = np.random.choice(neighbors, p=probs)
                            neighbors_set.add(next_node)
                            current_node = next_node
                        else:
                            break
                    else:
                        break
                
                if len(neighbors_set) >= max_neighbors:
                    break
            
            # Ограничение количества соседей
            sampled_neighbors = list(neighbors_set)[:max_neighbors]
            
            for neighbor in sampled_neighbors:
                sampled_edges.append([start_node, neighbor])
                weight = edge_weight_dict.get((start_node, neighbor), 1.0)
                sampled_weights.append(weight)
        
        if sampled_edges:
            sampled_edges = torch.tensor(sampled_edges, dtype=torch.long).t()
            sampled_weights = torch.tensor(sampled_weights, dtype=torch.float) if sampled_weights else None
        else:
            sampled_edges = torch.empty((2, 0), dtype=torch.long)
            sampled_weights = None
        
        return sampled_edges, sampled_weights
    
    def _importance_sampling(
        self,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        layer_idx: int,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Importance sampling на основе node features"""
        if node_features is None:
            # Fallback на uniform sampling
            return self._uniform_sampling(edge_index, node_indices, layer_idx, node_features, edge_weights)
        
        max_neighbors = self.config.neighbor_sizes[min(layer_idx, len(self.config.neighbor_sizes) - 1)]
        
        # Adjacency list
        adj_list = defaultdict(list)
        edge_weight_dict = {}
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
            if edge_weights is not None:
                edge_weight_dict[(src, dst)] = edge_weights[i].item()
        
        sampled_edges = []
        sampled_weights = []
        
        for node in node_indices:
            node = node.item()
            neighbors = adj_list[node]
            
            if len(neighbors) <= max_neighbors:
                sampled_neighbors = neighbors
            else:
                # Importance sampling на основе similarity node features
                node_feat = node_features[node]
                neighbor_feats = node_features[neighbors]
                
                # Cosine similarity
                similarities = F.cosine_similarity(
                    node_feat.unsqueeze(0), 
                    neighbor_feats, 
                    dim=1
                )
                
                # Softmax для превращения в вероятности
                probs = F.softmax(similarities / self.config.temperature, dim=0)
                
                # Sampling по вероятностям
                sampled_indices = torch.multinomial(probs, max_neighbors, replacement=False)
                sampled_neighbors = [neighbors[i] for i in sampled_indices]
            
            # Добавление рёбер
            for neighbor in sampled_neighbors:
                sampled_edges.append([node, neighbor])
                weight = edge_weight_dict.get((node, neighbor), 1.0)
                sampled_weights.append(weight)
        
        if sampled_edges:
            sampled_edges = torch.tensor(sampled_edges, dtype=torch.long).t()
            sampled_weights = torch.tensor(sampled_weights, dtype=torch.float) if sampled_weights else None
        else:
            sampled_edges = torch.empty((2, 0), dtype=torch.long)
            sampled_weights = None
        
        return sampled_edges, sampled_weights
    
    def _crypto_correlation_sampling(
        self,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        layer_idx: int,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Специализированный sampling для криптоактивов на основе корреляций"""
        # Приоритет высококоррелированных активов
        return self._importance_sampling(edge_index, node_indices, layer_idx, node_features, edge_weights)

class CryptoAggregator(nn.Module):
    """
    Кастомный aggregator для криптоданных
    
    Domain-Specific Aggregation
    """
    
    def __init__(self, input_dim: int, aggregation_type: str = 'mean', use_attention: bool = False):
        super().__init__()
        self.aggregation_type = aggregation_type
        self.use_attention = use_attention
        self.input_dim = input_dim
        
        if aggregation_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        elif aggregation_type == 'pool':
            self.pool_layers = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Агрегация neighborhood features
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            torch.Tensor: Aggregated features
        """
        row, col = edge_index
        
        if self.aggregation_type == 'mean':
            # Mean aggregation
            aggregated = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = col[row == i]
                if len(neighbors) > 0:
                    neighbor_features = x[neighbors]
                    
                    if self.use_attention:
                        attention_weights = self.attention(neighbor_features)
                        aggregated[i] = torch.sum(neighbor_features * attention_weights, dim=0) / torch.sum(attention_weights)
                    else:
                        aggregated[i] = torch.mean(neighbor_features, dim=0)
        
        elif self.aggregation_type == 'max':
            # Max aggregation
            aggregated = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = col[row == i]
                if len(neighbors) > 0:
                    neighbor_features = x[neighbors]
                    aggregated[i] = torch.max(neighbor_features, dim=0)[0]
        
        elif self.aggregation_type == 'lstm':
            # LSTM aggregation
            aggregated = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = col[row == i]
                if len(neighbors) > 0:
                    neighbor_features = x[neighbors].unsqueeze(0)  # [1, num_neighbors, dim]
                    _, (hidden, _) = self.lstm(neighbor_features)
                    aggregated[i] = hidden.squeeze(0).squeeze(0)
        
        elif self.aggregation_type == 'pool':
            # Pooling aggregation
            aggregated = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = col[row == i]
                if len(neighbors) > 0:
                    neighbor_features = x[neighbors]
                    pooled = self.pool_layers(neighbor_features)
                    aggregated[i] = torch.max(pooled, dim=0)[0]
        
        else:
            raise ValueError(f"Неизвестный тип агрегации: {self.aggregation_type}")
        
        return aggregated

class GraphSAGE(nn.Module):
    """
    Production-Ready GraphSAGE для crypto trading
    
    Scalable Graph Learning Architecture
    """
    
    def __init__(self, config: GraphSAGEConfig):
        super().__init__()
        self.config = config
        
        self._validate_config()
        self._build_layers()
        self._initialize_weights()
        
        # Sampler для neighborhood sampling
        self.sampler = AdvancedSampler(config)
        
        logger.info(f"Инициализирована GraphSAGE с {config.num_layers} слоями")
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if self.config.input_dim <= 0:
            raise ValueError("input_dim должно быть положительным")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim должно быть положительным")
        if not 0.0 <= self.config.dropout_rate <= 1.0:
            raise ValueError("dropout_rate должен быть в [0, 1]")
    
    def _build_layers(self) -> None:
        """Построение GraphSAGE архитектуры"""
        self.sage_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        
        # Размерности слоёв
        all_dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        
        # Создание SAGE слоёв
        for i in range(len(all_dims) - 1):
            # SAGEConv layer
            sage_layer = SAGEConv(
                in_channels=all_dims[i],
                out_channels=all_dims[i + 1],
                normalize=self.config.normalize,
                aggr=self.config.aggregation
            )
            self.sage_layers.append(sage_layer)
            
            # Custom aggregator для улучшенной агрегации
            aggregator = CryptoAggregator(
                input_dim=all_dims[i],
                aggregation_type=self.config.aggregation,
                use_attention=self.config.use_attention
            )
            self.aggregators.append(aggregator)
            
            # Batch normalization
            if self.config.use_batch_norm and i < len(all_dims) - 2:
                self.batch_norms.append(BatchNorm(all_dims[i + 1]))
        
        # Dropout и activation
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.activation = self._get_activation()
        
        # Output layer для финального предсказания
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim),
            nn.BatchNorm1d(self.config.output_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(self.config.output_dim, 1)
        )
    
    def _get_activation(self) -> nn.Module:
        """Получение функции активации"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def _initialize_weights(self) -> None:
        """Инициализация весов сети"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, SAGEConv)):
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, data: Data, use_sampling: bool = True) -> torch.Tensor:
        """
        Forward pass через GraphSAGE
        
        Args:
            data: PyG Data object
            use_sampling: Использовать sampling для больших графов
            
        Returns:
            torch.Tensor: Predictions
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        batch = getattr(data, 'batch', None)
        
        # Residual connections
        residual_x = x if self.config.use_residual else None
        
        # Проход через SAGE слои
        for i, (sage_layer, aggregator) in enumerate(zip(self.sage_layers[:-1], self.aggregators[:-1])):
            
            # Neighborhood sampling для больших графов
            if use_sampling and x.size(0) > 1000:
                node_indices = torch.arange(x.size(0))
                sampled_edges, sampled_weights = self.sampler.sample_neighbors(
                    edge_index, node_indices, i, x, edge_weight
                )
                if sampled_edges.size(1) > 0:
                    edge_index_layer = sampled_edges
                    edge_weight_layer = sampled_weights
                else:
                    edge_index_layer = edge_index
                    edge_weight_layer = edge_weight
            else:
                edge_index_layer = edge_index
                edge_weight_layer = edge_weight
            
            # SAGE convolution
            x = sage_layer(x, edge_index_layer)
            
            # Batch normalization
            if self.config.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Residual connection
            if (self.config.use_residual and residual_x is not None 
                and x.shape[-1] == residual_x.shape[-1]):
                x = x + residual_x
                residual_x = x
            
            # Dropout
            x = self.dropout(x)
        
        # Последний слой
        x = self.sage_layers[-1](x, edge_index)
        
        # Graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Финальное предсказание
        output = self.output_layer(x)
        
        return output
    
    def inductive_inference(self, data: Data) -> torch.Tensor:
        """
        Inductive inference для новых узлов
        
        Главное преимущество GraphSAGE - способность обрабатывать новые узлы
        """
        return self.forward(data, use_sampling=True)
    
    def get_node_embeddings(self, data: Data, layer_idx: int = -2) -> torch.Tensor:
        """Получение node embeddings из указанного слоя"""
        x, edge_index = data.x, data.edge_index
        
        # Проход до указанного слоя
        for i in range(min(len(self.sage_layers), layer_idx + 1)):
            x = self.sage_layers[i](x, edge_index)
            
            if i < len(self.sage_layers) - 1:
                if self.config.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        
        return x

class CryptoGraphSAGETrainer:
    """
    Специализированный тренер для GraphSAGE в crypto trading
    
    Scalable Training Pipeline
    """
    
    def __init__(self, model: GraphSAGE, config: GraphSAGEConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=15,
            verbose=True
        )
        
        self.model.to(self.device)
        
        # Метрики
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [],
            'sampling_efficiency': []
        }
        
        logger.info(f"GraphSAGE тренер готов на устройстве: {self.device}")
    
    def train_step(self, batch: Data) -> Dict[str, float]:
        """Шаг обучения с sampling optimization"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass с adaptive sampling
        predictions = self.model(batch, use_sampling=True)
        targets = batch.y.view(-1, 1).float()
        
        # Loss computation
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        
        # Backward pass
        mse_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
        
        self.optimizer.step()
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item()
        }
    
    def validate_step(self, batch: Data) -> Dict[str, float]:
        """Валидация с анализом sampling efficiency"""
        self.model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch, use_sampling=False)  # Full graph для validation
            targets = batch.y.view(-1, 1).float()
            
            mse_loss = F.mse_loss(predictions, targets)
            mae_loss = F.l1_loss(predictions, targets)
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item()
        }
    
    def predict(self, data: Union[Data, List[Data]], inductive: bool = False) -> np.ndarray:
        """
        Предсказание с поддержкой inductive learning
        
        Args:
            data: Входные данные
            inductive: Использовать inductive inference для новых узлов
        """
        self.model.eval()
        
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            if inductive:
                predictions = self.model.inductive_inference(batch)
            else:
                predictions = self.model(batch, use_sampling=False)
        
        return predictions.cpu().numpy()
    
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Обучение эпохи с sampling analysis"""
        train_metrics = {'loss': [], 'mae': []}
        
        for batch in train_loader:
            metrics = self.train_step(batch)
            for key in train_metrics:
                if key in metrics:
                    train_metrics[key].append(metrics[key])
        
        epoch_metrics = {f'train_{k}': np.mean(v) for k, v in train_metrics.items()}
        
        # Валидация
        if val_loader:
            val_metrics = {'loss': [], 'mae': []}
            
            for batch in val_loader:
                metrics = self.validate_step(batch)
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
            
            epoch_metrics.update({f'val_{k}': np.mean(v) for k, v in val_metrics.items()})
            
            # Обновление learning rate
            self.scheduler.step(epoch_metrics['val_loss'])
        
        # Сохранение истории
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return epoch_metrics
    
    def save_model(self, filepath: str) -> None:
        """Сохранение GraphSAGE модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"GraphSAGE модель сохранена в {filepath}")

def create_crypto_graphsage_model(
    input_dim: int,
    output_dim: int = 1,
    hidden_dims: Optional[List[int]] = None,
    neighbor_sizes: Optional[List[int]] = None,
    **kwargs
) -> Tuple[GraphSAGE, CryptoGraphSAGETrainer]:
    """
    Factory функция для создания GraphSAGE модели
    
    Factory Pattern with Configuration Injection
    """
    config = GraphSAGEConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        neighbor_sizes=neighbor_sizes,
        **kwargs
    )
    
    model = GraphSAGE(config)
    trainer = CryptoGraphSAGETrainer(model, config)
    
    return model, trainer

# Экспорт для использования
__all__ = [
    'GraphSAGE',
    'GraphSAGEConfig',
    'CryptoGraphSAGETrainer', 
    'AdvancedSampler',
    'CryptoAggregator',
    'create_crypto_graphsage_model'
]