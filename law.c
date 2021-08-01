#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#define REQUIRE_VELOCITY_DEPENDENCE


// Utilities

const static int customRandMax = 32767;

int customRand()
{
  static unsigned long seed = 1;
  seed = seed * 1103515245 + 12345;
  return (unsigned int)(seed / 65536) % 32768;
}


static double wrap(double phi)
{
    while (phi >  M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;
    return phi;
}


// Virtual machine

// Each instruction is 4 bits:
// 0..5   - push constant 0..5
// 6..9   - push argument a..d
// 10..13 - add, subtract, multiply, divide
// 14     - square
// 15     - no operation

static double eval(uint64_t func, const double *arg)
{
    double stack[16] = {0};
    int sp = -1;

    for (int i = 0; i < 16; i++)
    {
        const unsigned char instr = func & 15;
        func >>= 4;

        if (instr < 6)
            stack[++sp] = instr;
        else if (instr < 10)
            stack[++sp] = arg[instr - 6];
        else if (instr < 14)
        {
            if (sp < 1)
                return NAN;

            const double y = stack[sp--];
            switch (instr)
            {
                case 10: stack[sp] += y; break;
                case 11: stack[sp] -= y; break;
                case 12: stack[sp] *= y; break;
                case 13: if (y == 0) return NAN; stack[sp] /= y; break;
                default: break;
            }
        }
        else if (instr == 14)
        {
            if (sp < 0)
                return NAN;
            stack[sp] *= stack[sp];
        }
        else if (instr == 15)
        {
        }
        else
            assert(false);
    }

    if (sp != 0)
        return NAN;

    return stack[sp];
}


static double diff(uint64_t func, const double *arg, const int argIndex)
{
    const double dArg = 1e-9;
    double newArg[] = {arg[0], arg[1], arg[2], arg[3]};
    newArg[argIndex] += dArg;

    const double val = eval(func, arg);
    if (isnan(val))
        return NAN;

    const double newVal = eval(func, newArg);
    if (isnan(newVal))
        return NAN;

    return (newVal - val) / dArg;
}


static uint64_t encode(const char *str)
{
    assert(str);
    uint64_t func = 0;

    for (int i = 15; i >= 0; i--)
    {
        unsigned char instr = 0;
        const char c = str[i];
        assert(c != 0);

        if (c >= '0' && c <= '5')
            instr = c - '0';
        else if (c >= 'a' && c <= 'd')
            instr = c - 'a' + 6;
        else
        {
            switch (c)
            {
                case '+': instr = 10; break;
                case '-': instr = 11; break;
                case '*': instr = 12; break;
                case '/': instr = 13; break;
                case '^': instr = 14; break;
                case '.': instr = 15; break;
                default: assert(false); break;
            }
        }

        func |= instr;
        if (i > 0)
            func <<= 4;
    }
    return func;
}


static char *decode(uint64_t func, char *str)
{
    assert(str);

    for (int i = 0; i < 16; i++)
    {
        const unsigned char instr = func & 15;
        func >>= 4;

        if (instr < 6)
            str[i] = '0' + instr;
        else if (instr < 10)
            str[i] = 'a' + instr - 6;
        else
        {
            switch (instr)
            {
                case 10: str[i] = '+'; break;
                case 11: str[i] = '-'; break;
                case 12: str[i] = '*'; break;
                case 13: str[i] = '/'; break;
                case 14: str[i] = '^'; break;
                case 15: str[i] = '.'; break;
                default: assert(false); break;
            }
        }
    }
    str[16] = 0;
    return str;
}


static bool valid(uint64_t func)
{
    const double arg[] = {1.0, 2.0, 3.0, 4.0};
    return !isnan(eval(func, arg));
}


static uint64_t getRandom(void)
{
    uint64_t res = 0;
    for (int i = 0; i < 64; i += 15)
        res = (res << 15) | customRand();
    return res;
}


static uint64_t getRandomValid(void)
{
    while (1)
    {
        uint64_t func = getRandom();
        if (valid(func))
            return func;
    }
}


static uint64_t mutate(uint64_t func)
{
    const unsigned char pos = customRand() & 15;
    const unsigned char instr = customRand() & 15;
    return (func & ~(15 << (4 * pos))) | (instr << (4 * pos));
}


static uint64_t mutateValid(uint64_t func)
{
    func = mutate(func);
    if (valid(func))
        return func;
    return getRandom();
}


// Simulator

typedef struct
{
    double x, y, vx, vy;
} StateCart;


typedef struct
{
    double r, phi, rDot, phiDot;
} StatePolar;


static void update(StateCart *s, double GM, double dt)
{
    const double r = sqrt(s->x * s->x + s->y * s->y),
                 F = -GM / (r * r);

    s->vx += F * s->x / r * dt;
    s->vy += F * s->y / r * dt;

    s->x += s->vx * dt;
    s->y += s->vy * dt;
}


static void simulate(StatePolar *sp, int size, double dt)
{
    assert(size > 1);

    const double GM = 1.0;
    StateCart s = {.x = 10.0, .vy = 0.2};

    for (int i = 0; i < size; i++)
    {
        sp[i].r = sqrt(s.x * s.x + s.y * s.y);
        sp[i].phi = atan2(s.y, s.x);
        update(&s, GM, dt);
    }

    for (int i = 0; i < size - 1; i++)
    {
        sp[i].rDot = (sp[i + 1].r - sp[i].r) / dt;
        sp[i].phiDot = wrap(sp[i + 1].phi - sp[i].phi) / dt;
    }
}


// Optimizer

static double score(uint64_t func, StatePolar *sp, int size, double dt, double *gradRMS)
{
    const int decim = 1000;

    double conservedPrev = 0;
    double sumResSq = 0;

    for (int j = 0; j < 4; j++)
        gradRMS[j] = 0;

    for (int i = 0; i < size - 1; i += decim)
    {
        double arg[] = {sp[i].r, sp[i].phi, sp[i].rDot, sp[i].phiDot};

        const double conserved = eval(func, arg);
        if (isnan(conserved))
            return NAN;

        if (i > 0)
        {
            double residual = (conserved - conservedPrev) / (dt * decim);
            sumResSq += residual * residual;

            for (int j = 0; j < 4; j++)
            {
                const double grad = diff(func, arg, j);
                gradRMS[j] += grad * grad;
            }
        }

        conservedPrev = conserved;
    }

    if (sumResSq == 0)
        return NAN;

    for (int j = 0; j < 4; j++)
        gradRMS[j] = sqrt(gradRMS[j] / (size / decim));

    return -log10(sqrt(sumResSq / (size / decim)));
}


static bool gradValid(double *gradRMS)
{
    // Avoid trivial solutions
#ifdef REQUIRE_VELOCITY_DEPENDENCE
    return gradRMS[0] + gradRMS[1] > 0.001 && gradRMS[2] > 0.001 && gradRMS[3] > 0.001;
#else
    return gradRMS[0] > 0.001;
#endif
}


// Main program

static void printBest(uint64_t *population, int popSize, StatePolar *sp, int size, double dt, double threshold)
{
    for (int i = 0; i < popSize; i++)
    {
        double curGradRMS[4];
        const double curScore = score(population[i], sp, size, dt, curGradRMS);

        if (curScore > threshold && gradValid(curGradRMS))
        {
            char buf[17];
            printf("Script: %s   score: %10.5lf   grad: %10.5lf %10.5lf %10.5lf %10.5lf\n",
                   decode(population[i], buf), curScore,
                   curGradRMS[0], curGradRMS[1], curGradRMS[2], curGradRMS[3]);            
        }
    }
}

int main()
{
    // Prepare simulated data
    const int size = 100000;
    const double dt = 0.001;

    StatePolar *states = malloc(size * sizeof(StatePolar));
    simulate(states, size, dt);

    // Generate random initial candidates
    const int popSize = 500;
    uint64_t population[popSize];

    for (int i = 0; i < popSize; i++)
        population[i] = getRandomValid();

    const double threshold = 5;
    printf("Best of initial population:\n");
    printBest(population, popSize, states, size, dt, threshold);    

    // Optimize by simulated annealing
    for (int generation = 0; generation < 1000; generation++)
    {
        printf("Generation %d\n", generation);

        for (int i = 0; i < popSize; i++)
        {
            double curGradRMS[4];
            const double curScore = score(population[i], states, size, dt, curGradRMS);

            if (isnan(curScore) || !gradValid(curGradRMS) || curScore < 0)
            {
                population[i] = getRandomValid();
                continue;
            }

            const uint64_t mutated = mutateValid(population[i]);

            double mutGradRMS[4];
            const double mutScore = score(mutated, states, size, dt, mutGradRMS);

            if (isnan(mutScore) || !gradValid(mutGradRMS))
                continue;

            const double temperature = 0.2;
            const double prob = (mutScore > curScore) ? 1.0 : (exp((mutScore - curScore) / temperature));

            if (customRand() < prob * customRandMax)
                population[i] = mutated;
        }
    }

    printf("Best of results:\n\n");
    printBest(population, popSize, states, size, dt, threshold);

    free(states);
    return 0;
}


