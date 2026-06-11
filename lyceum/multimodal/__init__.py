"""Multimodal (vision) capability for Lyceum.

Frontier models are *natively multimodal*: a non-text input (an image, an audio
clip) is passed through a modality-specific **encoder** that projects it into the
**same token-embedding space** the language model already uses for text. Once an
image has been turned into a short sequence of "image tokens", the transformer
treats them no differently from word tokens -- the text simply *attends* to the
image as a prefix in the context window.

This package builds a scaled-down but *real* version of that mechanism on top of
the existing :class:`~lyceum.model.transformer.LyceumLM`:

  * :func:`~lyceum.multimodal.vision.image_to_patches` -- ViT-style patchify.
  * :class:`~lyceum.multimodal.vision.PatchEmbedder` -- a tiny linear projector
    (stand-in for a full ViT/CLIP encoder) that maps pixel patches into the
    language model's ``dim``-dimensional embedding space, plus learned position
    embeddings, producing image "tokens".
  * :class:`~lyceum.multimodal.vision.MultimodalLM` -- wraps the LM + projector
    and runs the transformer trunk over ``[image tokens] ++ [text tokens]``.

Run the self-test with::

    python -m lyceum.multimodal.vision
"""
