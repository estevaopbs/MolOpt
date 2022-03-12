from MolOpt.MolOpt import MolOpt, Molpro_Molecule, Create, Mutate, Crossover, Strategies


"""Collection of tests
"""


def Al10() -> MolOpt:
    """Al10 optimization test

    Returns
    -------
    MolOpt
        Molecular optimization object
    """
    create_methods = Create([MolOpt.randomize, MolOpt.mutate_first, MolOpt.mutate_best], [1, 0, 0])
    mutate_methods = Mutate([MolOpt.particle_permutation, MolOpt.piece_displacement, MolOpt.particle_displacement,
        MolOpt.piece_rotation, MolOpt.piece_reflection_replace_original, MolOpt.piece_reflection_replace_opposite,
        MolOpt.enlarge, MolOpt.reduce], [0, 1, 1, 1, 1, 1, 1, 1])
    crossover_methods = Crossover([MolOpt.piece_crossover,], [1])
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    optimization = MolOpt(
        first_molecule = Molpro_Molecule.load('MolOpt/tests/inputs/Al10.inp'),
        displacement_range = (-1, 1),
        rotation_range = (-30, 30),
        piece_size_range = (2, 5),
        distance_range = (1, 5),
        enlarge_reduce_range = (0.7, 1.3),
        strategies = strategies,
        max_age = 10,
        pool_size = 16,
        mutate_after_crossover = False,
        crossover_elitism = None,
        elitism_rate = None,
        freedom_rate = 1,
        parallelism = True,
        aways_local_opt = False,
        time_toler_opt = None,
        gens_toler_opt = None,
        max_seconds = None,
        time_toler = None,
        gens_toler = None,
        max_gens = 50,
        max_fitness = None,
        save_directory = None,
        adapting_rate = None,
        adapting_args = None,
        threads_per_calc = 2
    )
    return optimization


def Al9Si() -> MolOpt:
    """Al9Si optimization test

    Returns
    -------
    MolOpt
        Molecular optimization object
    """
    create_methods = Create([MolOpt.randomize, MolOpt.mutate_first, MolOpt.mutate_best], [1, 0, 0])
    mutate_methods = Mutate([MolOpt.particle_permutation, MolOpt.piece_displacement, MolOpt.particle_displacement,
        MolOpt.piece_rotation, MolOpt.piece_reflection_replace_original, MolOpt.piece_reflection_replace_opposite,
        MolOpt.enlarge, MolOpt.reduce], [1, 1, 1, 1, 1, 1, 1, 1])
    crossover_methods = Crossover([MolOpt.piece_crossover], [1])
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    optimization = MolOpt(
        first_molecule = Molpro_Molecule.load('MolOpt/tests/inputs/Al9Si.inp'),
        displacement_range = (-1, 1),
        rotation_range = (-30, 30),
        piece_size_range = (2, 5),
        distance_range = (1, 5),
        enlarge_reduce_range = (0.7, 1.3),
        strategies = strategies,
        max_age = 10,
        pool_size = 16,
        mutate_after_crossover = False,
        crossover_elitism = None,
        elitism_rate = None,
        freedom_rate = 1,
        parallelism = True,
        aways_local_opt = False,
        time_toler_opt = None,
        gens_toler_opt = None,
        max_seconds = None,
        time_toler = None,
        gens_toler = None,
        max_gens = 50,
        max_fitness = None,
        save_directory = None,
        adapting_rate = None,
        adapting_args = None,
        threads_per_calc = 2
    )
    return optimization
