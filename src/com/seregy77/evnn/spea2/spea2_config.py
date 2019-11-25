class Spea2Config:
    _populationSize = None
    _archiveSize = None
    _maxIterations = None
    _crossoverProbability = None
    _mutationProbability = None

    def __init__(self,
                 population_size,
                 archive_size,
                 max_iterations=20,
                 crossover_probability=0.8,
                 mutation_probability=0.2):
        self._populationSize = population_size
        self._archiveSize = archive_size
        self._maxIterations = max_iterations
        self._crossoverProbability = crossover_probability
        self._mutationProbability = mutation_probability

    @property
    def population_size(self):
        return self._populationSize

    @property
    def archive_size(self):
        return self._archiveSize

    @property
    def max_iterations(self):
        return self._maxIterations

    @property
    def crossover_probability(self):
        return self._crossoverProbability

    @property
    def mutation_probability(self):
        return self._mutationProbability
