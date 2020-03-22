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
    alpha = trial.suggest_loguniform('alpha', 1E-6, 1E2)
    beta = trial.suggest_loguniform('beta', 1E-2, 1E4)
    gamma = trial.suggest_loguniform('gamma', 1E-6, 1E2)
    return squared_error(alpha, beta, gamma)


def fit():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=128)
    return study.best_params


def plot(params):
    observed = pandas.Series(
        I_observed,
        index=pandas.date_range('2020-03-04', periods=fit_days, freq='D'),
    )
    predict_duration = 60
    prediction = pandas.Series(
        list(StateIterator(days=predict_duration, **params)),
        index=pandas.date_range('2020-03-04', periods=predict_duration, freq='D'),
    )
    print(f'infected population: {int(prediction[-1])}')
    seaborn.set()
    ax = observed.plot.line(style='o', label='observed')
    prediction.plot.line(label='prediction')
    ax.legend()
    ax.set_yscale('log')
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('out.png')


if __name__ == '__main__':
    params = fit()
    plot(params)
