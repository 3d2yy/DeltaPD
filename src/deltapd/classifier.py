import numpy as np
from typing import Dict, Tuple

class PDClassifier:
    """
    Clasificador de Descargas Parciales basado en 5 indicadores predictivos:
    1. Weibull Beta
    2. Burstiness Index
    3. Coeficiente de Variación (CV) de Delta t
    4. Factor de Fano
    5. Entropía de Fase PRPD
    """
    
    def __init__(self):
        self.categories = ['Corona', 'Interna', 'Superficial', 'Ruido']
        # Definición de reglas por indicador (scoring)
        # Esto puede refinarse con la Fase 0 de calibración
        
    def classify(self, indicators: Dict[str, float]) -> Tuple[str, float, Dict[str, int]]:
        scores = {cat: 0 for cat in self.categories}
        
        beta = indicators.get('weibull_beta', 1.0)
        burst = indicators.get('burstiness', 0.0)
        cv = indicators.get('cv', 1.0)
        fano = indicators.get('fano', 1.0)
        entropy = indicators.get('entropy', 1.0)
        
        # 1. Weibull Beta Rules
        if beta > 2.0:      scores['Corona'] += 2
        elif beta < 0.8:    scores['Interna'] += 1
        elif 0.9 < beta < 1.1: scores['Ruido'] += 1
        
        # 2. Burstiness Rules
        if burst > 0.2:     scores['Interna'] += 1
        elif burst < -0.2:  scores['Corona'] += 1
        
        # 3. CV Rules
        if cv > 1.5:        scores['Interna'] += 1
        elif cv < 0.6:      scores['Corona'] += 1
        
        # 4. Fano Factor Rules
        if fano > 10.0:     scores['Superficial'] += 2
        elif fano > 3.0:    scores['Interna'] += 1
        
        # 5. Phase Entropy Rules
        if entropy > 0.9:   scores['Corona'] += 1   # High entropy = spread
        elif entropy < 0.5: scores['Interna'] += 1  # Low entropy = concentrated
        elif 0.5 <= entropy <= 0.8: scores['Superficial'] += 1
        
        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[winner] / total if total > 0 else 0
        
        return winner, confidence, scores

def classify_events(df_indicators: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clasifica una serie de indicadores y devuelve las categorías y confianzas.
    """
    clf = PDClassifier()
    n = len(next(iter(df_indicators.values())))
    categories = []
    confidences = []
    
    for i in range(n):
        current = {k: v[i] for k, v in df_indicators.items()}
        # Ignorar si hay muchos NaNs
        if np.isnan(list(current.values())).sum() > 2:
            categories.append('Unknown')
            confidences.append(0.0)
            continue
            
        cat, conf, _ = clf.classify(current)
        categories.append(cat)
        confidences.append(conf)
        
    return np.array(categories), np.array(confidences)
