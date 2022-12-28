import argparse
from model.deposit_evolution import DepositEvolution

boundary_conditions = {
    "upstream_slope": None,
    "flow_scale": None,
    "upstream_gsd": {
        2: None,
        1: None,
        0: None,
        -1: None,
        -2: None,
        -2.5: None,
        -3: None,
        -3.5: None,
        -4: None,
        -4.5: None,
        -5: None,
        -5.5: None,
        -6: None,
        -7: None,
        -7.5: None,
        -8: None,
        -8.5: None,
        -9: None
    }
}

reaches = {
    "timestep": None,
    "deposit_mass": None,
    "reaches": {
        "upstream": {
        "flow_scale": None,
        "elevation": None,
        "width": None,
        "height": None,
        "gsd": {
            2: None,
            1: None,
            0: None,
            -1: None,
            -2: None,
            -2.5: None,
            -3: None,
            -3.5: None,
            -4: None,
            -4.5: None,
            -5: None,
            -5.5: None,
            -6: None,
            -7: None,
            -7.5: None,
            -8: None,
            -8.5: None,
            -9: None
        },
        "Qs_in": None,
        "Qs_out": None
    },
        "deposit_upstream": {
        "flow_scale": None,
        "elevation": None,
        "width": None,
        "height": None,
        "gsd": {
            2: None,
            1: None,
            0: None,
            -1: None,
            -2: None,
            -2.5: None,
            -3: None,
            -3.5: None,
            -4: None,
            -4.5: None,
            -5: None,
            -5.5: None,
            -6: None,
            -7: None,
            -7.5: None,
            -8: None,
            -8.5: None,
            -9: None
        },
        "Qs_in": None,
        "Qs_out_bed": None,
        "Qs_out_bank": None,
    },
        "deposit_downstream": {
        "flow_scale": None,
        "elevation": None,
        "width": None,
        "height": None,
        "gsd": {
            2: None,
            1: None,
            0: None,
            -1: None,
            -2: None,
            -2.5: None,
            -3: None,
            -3.5: None,
            -4: None,
            -4.5: None,
            -5: None,
            -5.5: None,
            -6: None,
            -7: None,
            -7.5: None,
            -8: None,
            -8.5: None,
            -9: None
        },
        "Qs_in": None,
        "Qs_out_bed": None,
        "Qs_out_bank": None
    },
        "downstream": {
        "flow_scale": None,
        "elevation": None,
        "width": None,
        "height": None,
        "gsd": {
            2: None,
            1: None,
            0: None,
            -1: None,
            -2: None,
            -2.5: None,
            -3: None,
            -3.5: None,
            -4: None,
            -4.5: None,
            -5: None,
            -5.5: None,
            -6: None,
            -7: None,
            -7.5: None,
            -8: None,
            -8.5: None,
            -9: None
        },
        "Qs_in": None,
        "Qs_out": None
    }
    }
}

def main():

    parser = argparse.ArgumentParser()


if __name__ == '__main__':
    main()