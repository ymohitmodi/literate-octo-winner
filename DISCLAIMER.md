# Disclaimer

**Lyceum is an educational reconstruction, not a reproduction.**

This project was built to *learn* and *teach* how a modern frontier language
model and its surrounding platform could be designed — from data and
tokenization through training, alignment, inference, memory, serving, security,
and evaluation. It is assembled from a first-principles understanding of each
individual component, scaled down so the whole lifecycle runs on a single
CPU-only machine (and lights up additional features when a GPU is present).

Please read the following carefully:

1. **It simulates understanding, not reality.** Every component here is *one
   plausible way* a piece could work, implemented small for clarity. It is an
   exercise in understanding how the parts fit together — **it is not an
   assurance, specification, or claim about how any real system actually works
   internally.**

2. **"Mythos" is a teaching codename, not this project.** The Frontier Model
   Field Manual that inspired this work refers to its hypothetical model as
   "Mythos." This project is independently named **Lyceum** and is not that
   model, nor any real product. References to "Mythos" in the documentation
   point to the manual's teaching device only.

3. **No affiliation or reverse-engineering.** Lyceum does not reproduce, reverse
   engineer, or reveal the weights, data, architecture, or training process of
   Claude (Anthropic) or any other commercial model. It draws only on publicly
   described, general machine-learning methods. Any resemblance to a specific
   product is incidental to those shared public methods.

4. **Not production-grade.** The model is tiny and its text quality is
   rudimentary; the point is that the *mechanisms* are real and inspectable, not
   that the outputs are good. The security controls demonstrate concepts and
   must not be relied upon to secure a real system without proper review,
   hardening, and the secrets/configuration changes noted in `docs/SECURITY.md`.

5. **Security material is for defensive learning.** The included attack
   demonstrations (data poisoning, membership inference, pickle RCE, prompt
   injection, etc.) exist to teach defenders how the corresponding controls work.
   They run only against this project's own toy artifacts. Use them only for
   authorized, educational, defensive purposes.

6. **Use at your own risk.** This is provided as-is, without warranty of any
   kind, for educational use.

If you build on Lyceum, please preserve this disclaimer.
