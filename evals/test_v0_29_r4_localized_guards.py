from semantic_relevance_facet_judge import _q04_r4_has_directional_departure_anchor

def main():
    assert _q04_r4_has_directional_departure_anchor("Paulo and Cristina take off on a forty day adventure into the sometimes dangerous Mojave Desert.")
    assert _q04_r4_has_directional_departure_anchor("They took off on an expedition toward the mountains.")
    assert _q04_r4_has_directional_departure_anchor("She takes off on the journey through the desert.")
    assert not _q04_r4_has_directional_departure_anchor("Sales took off after the product launch.")
    assert not _q04_r4_has_directional_departure_anchor("He took off his coat before dinner.")
    assert not _q04_r4_has_directional_departure_anchor("The story takes off after the first chapter.")
    assert not _q04_r4_has_directional_departure_anchor("Events unfold over three years and end in violent explosive action.")
    print("v0.29 r4 localized Q04 directional-departure tests passed")
if __name__ == '__main__': main()
