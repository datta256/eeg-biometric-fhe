# Can We Compare EEG Biometrics Without Exposing the Template?

*An experimental prototype combining EEG embeddings with CKKS homomorphic encryption—and what it does and does not prove.*

Biometrics have an awkward property: the thing that makes them useful for authentication also makes them sensitive. A password can be rotated after a breach. A face, fingerprint, or neural signal cannot simply be reissued in the same way.

I have been exploring a small piece of a possible alternative: **could a system compare EEG-derived biometric templates while keeping the templates encrypted during the similarity calculation?**

This project is an experiment, not a finished authentication product. The goal is to make the engineering question concrete, measure the arithmetic, and be explicit about the parts that remain unsolved.

## The basic idea

The prototype starts with a pre-recorded SSVEP EEG trial. It selects eight posterior channels—POZ, PO3, PO4, PO7, PO8, OZ, O1 and O2—and normalizes each channel over its time window. A compact temporal-spatial encoder maps the trial to a 32-dimensional vector.

At enrollment, a system would keep a reference representation. During verification, it would produce a second representation and compare the two. In this experiment, similarity is measured with a dot product or cosine similarity, depending on the evaluation path.

The privacy question is what happens at the comparison step. Instead of passing two plaintext vectors to the arithmetic operation, the FHE demonstration encrypts the embeddings using CKKS, computes the dot product over ciphertexts, and decrypts the result to compare it with the plaintext score.

CKKS supports approximate arithmetic on encrypted values. That makes it useful for numerical workloads, but it also means we must measure approximation error rather than assume encrypted and plaintext results are bit-for-bit identical.

## What I implemented

The repository brings together four pieces:

1. **An EEG encoder and evaluation scripts** for cross-session verification experiments.
2. **A FastAPI service** with a plaintext baseline and a CKKS demonstration endpoint.
3. **A browser interface** that lets a user select a pair of pre-recorded trials and compare the two paths.
4. **Validation scripts and result artifacts** for examining multiple frequencies and session pairs.

The encryption parameters used in the demonstration are a polynomial modulus degree of 8192, coefficient modulus bit sizes `[60, 40, 40, 60]`, and a scale of `2**40` through TenSEAL.

The demo is intentionally narrow. It demonstrates encrypted similarity arithmetic—not end-to-end encrypted EEG inference.

## What the experiment showed

In the recorded multi-condition run, the evaluation covered nine stimulation frequencies from 8 to 12 Hz in 0.5 Hz increments, across ten unique session-pair combinations: 90 conditions in total. The evaluation used identities held out from the checkpoint's recorded training identities, and sampled impostor comparisons for the FHE arithmetic check.

The saved summary reported:

- Mean AUC across conditions: **0.9882**
- Mean descriptive EER across conditions: **4.89%**
- Encrypted pair evaluations: **21,600**
- Mean absolute plaintext-versus-FHE score error: **2.412 × 10⁻⁶**
- Maximum absolute score error: **2.609 × 10⁻⁶**
- Decision flips among tested FHE pairs: **0**

The last three figures are the central result for the homomorphic-encryption part: for the tested vectors and parameter set, the encrypted calculation produced scores very close to the plaintext calculation, and the tested decisions did not change.

But the numbers need context. Subjects and session/frequency conditions recur, so these observations are not all independent samples. The threshold was not independently calibrated for every condition, and some conditions show much higher false-reject rates than the average suggests. The results are a snapshot from this experiment, not a universal estimate of biometric performance.

## The most important caveat: encryption is not the whole system

It would be misleading to say that this prototype keeps the entire biometric private.

**First, the encoder runs in plaintext.** The API loads the EEG trial and performs model inference before encrypting the resulting embedding. The raw trial and the intermediate computation are therefore not protected from the API operator by CKKS in this version.

**Second, the demo API holds the secret key.** The endpoint decrypts the encrypted score inside the same process so it can report the result and compare numerical error. This is a convenient way to test the math; it is not a distributed trust model.

**Third, a similarity score can itself be sensitive.** A real protocol should avoid revealing more than necessary. Ideally, the system would expose only the decision needed by the application, rather than a reusable score, while carefully considering who can trigger comparisons and how often.

**Fourth, there is no ZKML proof here.** Numerical agreement between plaintext and encrypted arithmetic is not proof that a remote server ran the intended neural network, used the correct template, or followed the full protocol. Proof of model execution and proof or verification of encrypted arithmetic are separate problems.

These boundaries matter more than a demo label saying “encrypted.” Privacy depends on the complete data path, keys, threat model, and output—not one operation in isolation.

## What a stronger architecture would need

A more serious design would separate the following responsibilities:

- **Device or trusted acquisition boundary:** collect and preprocess EEG with a clearly specified trust model.
- **Model execution:** decide whether inference happens locally, inside a trusted environment, or through a verifiable privacy-preserving protocol.
- **Encrypted comparison:** use CKKS or another suitable cryptographic method with parameters selected for the required precision and threat model.
- **Key management:** use a well-designed threshold protocol if no single service should hold the decryption key. Standard single-party TenSEAL contexts cannot simply be split into shares and assumed to work as a secure threshold system.
- **Decision policy:** reveal only the minimum result needed, potentially through a secure comparison protocol.
- **Verification and accountability:** specify what a proof actually establishes, protect against malicious validators, and record auditable outcomes without leaking biometric templates.

A smart contract cannot directly evaluate an arbitrary CKKS ciphertext by itself. An on-chain application would need a defined bridge—such as a proof, an attestation, or a threshold-signature protocol—and a clear account of what the bridge trusts.

## What I would test next

The next iteration should prioritize evaluation discipline and system boundaries over adding more cryptographic labels:

1. Freeze the model and split identities into separate training, development/calibration, and final-test groups.
2. Select the operating threshold only on development data, then report false-accept and false-reject rates on untouched identities.
3. Report subject-level confidence intervals and test across sessions and frequencies without treating repeated conditions as independent observations.
4. Benchmark encryption, encrypted computation, serialization, and key setup separately.
5. Design the threshold-key and decision protocol before integrating it into a wallet.
6. Investigate proof systems for neural-network inference separately from the encrypted dot-product operation.
7. Have the full protocol reviewed against an explicit adversarial threat model.

## Why I am publishing the prototype

The useful outcome is not “EEG identity is solved.” It is a reproducible starting point for a more specific question: **how accurately can this EEG representation compare across sessions, and how closely can CKKS reproduce the selected similarity operation on encrypted embeddings?**

The current experiment suggests that the encrypted dot-product arithmetic can closely track the plaintext score for the tested inputs. It also exposes the harder parts: calibration, generalization across conditions, private inference, threshold decryption, and trustworthy authorization.

I have included the demo, evaluation scripts, recorded artifacts, and the limitations in the repository so that the work can be inspected and challenged. If you reproduce it, I would be especially interested in independent results on the held-out protocol and practical approaches to secure threshold comparison.

**Repository:** [Add the public GitHub URL here after creating the repository.]

*Disclaimer: This is an experimental research prototype. It is not a medical device, a production biometric authenticator, or a secure wallet implementation. Do not use it to protect real funds or sensitive identities.*
