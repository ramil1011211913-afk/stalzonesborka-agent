"""Pure current-Wiki formulas. No GPU, database, network or builder imports."""
def calculate_effective_health(stats: dict) -> float:
    health = stats.get("health_bonus", 0.0)
    bullet = stats.get("bullet_dmg_factor", 0.0)
    return (health + 100.0) / 100.0 * (100.0 + bullet)

def calculate_healing_per_second(stats: dict) -> float:
    regen = stats.get("regeneration_bonus", 0.0)
    art_heal = stats.get("artefakt_heal", 0.0)
    heal_eff = stats.get("heal_efficiency", 0.0)
    return (regen + 2.5) / 5.0 + art_heal * (1.0 + heal_eff / 100.0)
