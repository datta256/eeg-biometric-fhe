import ezkl

print("Module:", ezkl.__file__)
print("Version:", getattr(ezkl, "__version__", "not exposed"))
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
    print(f"{name}: {hasattr(ezkl, name)}")
