import multiprocessing

import matplotlib.pyplot
import numpy
import optuna
import pandas
import seaborn

N0 = 39560000  # population
# fitting period: 03-04 to 03-20
I_observed = numpy.array([
    53, 60, 69, 88, 114, 133, 157, 177, 198, 247,
    335, 392, 472, 598, 675, 1006, 1224,
])
fit_days = I_observed.shape[0]

class State:
    def __init__(self, alpha, beta, gamma) -> None:
        # as of 2020-03-04
        self.I = 53
        self.S = N0 - self.I
        self.R = 0.0
        self.E = 0.0
        self.alpha = alpha
        self.beta = beta / N0
        self.gamma = gamma

    def update(self) -> None:
        dt = 1  # day
        bis = self.beta * self.I * self.S
        ae = self.alpha * self.E
        gi = self.gamma * self.I
        dS = -bis * dt
        dE = (bis - ae) * dt
        dI = (ae - gi) * dt
        dR = gi * dt

        self.S += dS
        self.E += dE
        self.I += dI
        self.R += dR

        self.S = max(0, min(N0, self.S))
        self.I = max(0, min(N0, self.I))

    def __list__(self):
        return (self.S, self.E, self.I, self.R)


class StateIterator:
    def __init__(self, days, alpha, beta, gamma):
        self._days = days
        self._state = State(alpha, beta, gamma)

    def __iter__(self):
        self._cnt = 0
        return self

    def __next__(self):
        infected = self._state.I
        self._state.update()
        if self._cnt < self._days:
            self._cnt += 1
            return infected
        else:
            raise StopIteration


def squared_error(alpha, beta, gamma) -> float:
    I_predicted = list(StateIterator(fit_days, alpha, beta, gamma))
    return numpy.sum((I_observed - I_predicted) ** 2)


def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1E-7, 1E-2)
    beta = trial.suggest_loguniform('beta', 1E-1, 1E6)
    gamma = trial.suggest_loguniform('gamma', 1E-9, 1E-2)
    return squared_error(alpha, beta, gamma)


def fit(seed):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=256)
    return study.best_params


def plot(params):
    observed = pandas.Series(
        I_observed,
        index=pandas.date_range('2020-03-04', periods=fit_days, freq='D'),
    )
    predict_duration = 90
    predictions = [
        pandas.Series(
            list(StateIterator(days=predict_duration, **param)),
            index=pandas.date_range('2020-03-04', periods=predict_duration, freq='D'),
        )
        for param in params
    ]
    seaborn.set()
    ax = None
    ax = seaborn.lineplot(
        data=pandas.concat(predictions),
        label='prediction',
        ax=ax,
    )
    ax = observed.plot.line(style='o', label='observed', ax=ax)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('date')
    ax.set_ylabel('infected')
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('out.png')


if __name__ == '__main__':
    pool = multiprocessing.Pool(8)
    params = pool.map(
        fit,
        range(64),
    )
    plot(params)
