"""Mechanistic interpretability tools for Lyceum.

The Frontier manual's interpretability chapter, shrunk to a CPU demo. The core
claims this package illustrates:

  * **Features and circuits** - a network's behavior decomposes into reusable
    *features* (directions in activation space) wired into *circuits*.
  * **Superposition** - models pack more features than they have neurons by
    storing them as overlapping linear combinations, so raw neurons are
    polysemantic and hard to read.
  * **Sparse autoencoders (SAEs)** - train an overcomplete, sparse dictionary
    that re-expresses the dense residual stream as a few active, more
    monosemantic features (see :mod:`lyceum.interpretability.sae`).
  * **Activation steering** - because features are directions, you can *add* a
    feature's direction back into the residual stream to steer generation.
"""
