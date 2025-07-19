#ifndef _ISING_H
#define _ISING_H
#include <cstdlib>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <array>
#include <omp.h>

class Ising
{
public:
    // constructor
    Ising(int L, float ext_t)
        : L(L), t(ext_t), energy_table({1.0f,1.0f,1.0f,std::exp(-4.0f / t), std::exp(-8.0f / t)})
    {
        spin = std::vector<int>(L * L, 1);
        e_data = std::vector<int>();
        m_data = std::vector<int>();
    }

    float get_temperature() const { return t; }

    void set_spin(const std::vector<int> &new_spin) { spin = new_spin; }
    std::vector<int> get_spin() const { return spin; }

    void set_data(const std::vector<int> &ext_m_data, const std::vector<int> &ext_e_data)
    {
        m_data = ext_m_data;
        e_data = ext_e_data;
    }

    std::vector<int> get_m_data() const { return m_data; }
    std::vector<int> get_e_data() const { return e_data; }

    void run(int Ntest, int spacing)
    {
        for (int i = 0; i < Ntest; i++)
        {
            sweep(spacing);
            record();
        }
    }

private:
    const int L;
    const float t;                   // temperature (non-const for updating)
    std::array<float,5> energy_table;
    std::vector<int> spin;           // spin configuration
    std::vector<int> e_data, m_data; // energy and magnetization data

    void sweep(int spacing)
    {
        for (int i = 0; i < spacing; i++)
        {
            checkboard(0);
            checkboard(1);
        }
    }

    void checkboard(bool is_even)
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int x = 0; x < L; x++)
        {
            for (int y = 0; y < L; y++)
            {
                if ((x + y) % 2 == is_even)
                {
                    if (x == 0 || x == L - 1 || y == 0 || y == L - 1)
                    {
                        local_update_bdr(x, y); // boundary condition
                    }
                    else
                    {
                        local_update(x, y); // interior
                    }
                }
            }
        }
    }

    void record()
    {
        int total_m = 0;
        int total_e = 0; // count of aligned bonds
        for (int x = 0; x < L; ++x)
        {
            for (int y = 0; y < L; ++y)
            {
                // Magnetization
                total_m += spin[ind(x, y)];
                total_e += (spin[ind(x, y)] * (spin[ind((x + 1) % L, y)] + spin[ind(x, (y + 1) % L)])); // right neighbor
            }
        }
        m_data.push_back(total_m);
        e_data.push_back(total_e);
    }

    void local_update_bdr(int x, int y)
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<float> dist(0.0, 1.0);

        int dE = spin[ind(x, y)] * (spin[ind((x + 1) % L, y)] + spin[ind((x - 1 + L) % L, y)] + spin[ind(x, (y + 1) % L)] + spin[ind(x, (y - 1 + L) % L)]); // right and left neighbors

        if (dist(rng) < energy_table[dE/2 + 2])
        {
            spin[ind(x, y)] *=-1;
            return;
        }
    }

    void local_update(int x, int y)
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<float> dist(0.0, 1.0);

        int dE = spin[ind(x, y)] * (spin[ind(x + 1, y)] + spin[ind(x - 1, y)] + spin[ind(x, y + 1)] + spin[ind(x, y - 1)]); // right and left neighbors

        if (dist(rng) < energy_table[dE/2 + 2])
        {
            spin[ind(x, y)] *=-1;
            return;
        }
    }

    inline unsigned int ind(int x, int y) const { return x * L + y; }
};
#endif