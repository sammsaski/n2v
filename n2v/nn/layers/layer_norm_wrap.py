"""LayerNorm wrapper (re-exports torch.nn.LayerNorm via subclass marker).

n2v's dispatcher uses ``isinstance(layer, nn.LayerNorm)`` directly, but
this thin subclass lets users distinguish n2v-aware LayerNorm usage from
ad-hoc usage when needed (e.g., for documentation grouping).
"""

import torch.nn as nn


class LayerNormWrap(nn.LayerNorm):
    """Marker subclass of :class:`torch.nn.LayerNorm`.

    Behaviour is identical to ``nn.LayerNorm``; existence is purely
    organisational for the wrapper package.
    """

    pass
