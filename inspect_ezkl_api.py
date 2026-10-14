import inspect
import ezkl

print("EZKL:", ezkl.__file__)
print("\n=== FUNCTION SIGNATURES ===")

for name in [
    "gen_settings",
    "calibrate_settings",
    "compile_circuit",
    "get_srs",
    "setup",
    "gen_witness",
    "prove",
    "verify",
]:
    fn = getattr(ezkl, name, None)
    if fn is None:
        print(f"\n{name}: MISSING")
        continue

    try:
        print(f"\n{name}{inspect.signature(fn)}")
    except Exception as e:
        print(f"\n{name}: signature unavailable ({e})")
        print("Doc:", inspect.getdoc(fn))

print("\n=== CALIBRATION DOC ===")
print(inspect.getdoc(ezkl.calibrate_settings))

print("\n=== SETTINGS DOC ===")
print(inspect.getdoc(ezkl.gen_settings))
