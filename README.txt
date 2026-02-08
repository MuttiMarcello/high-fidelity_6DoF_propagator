# High-Fidelity 6 Degrees-of-Freedom Propagator

Individual project on high-fidelity attitude and orbit propagator for LEO operations.

## Overview

This repository contains an ongoing project focused on high-fidelity 6DoF dynamics simulation in Low-Earth Orbit perturbative environment.

The objective is to implement and assess orbit propagation using two-body problem dynamics with J2, drag, solar radiation pressure, and third-body Moon perturbations, and attitude propagation using Euler rigid-body dynamics with gravity gradient and magnetic disturbance torques.

The work is currently under active development.

## Methodology

The project follows the given methodology:
- 2BP orbital dynamics and Euler rigid-body attitude dynamics.
- Gravitational orbital perturbation model considering J2 zonal harmonic.
- Atmospheric density model based on U.S. Standard Atmosphere (1976) for 0-25 km height, and CIRA-72 for > 25 km height.
- Magnetic disturbance torque model based on IGRF2000 geomagnetic reference field.

Relevant references and theoretical background are drawn from standard literature and research publications in the field.

## Current status

This project is a work in progress.

At the current stage:

- Core attitude and orbital dynamics are implemented
- J2 and third-body Moon perturbations are implemented
- Gravity gradient and magnetic disturbance torques are implemented

Planned next steps include:

- Validation of perturbative effects against reference solutions and external tools
- Implementation of drag and SRP orbital disturbance
- Implementation of drag and SRP attitude disturbance torques
- Change approach to Legendre polynomials to avoid recursivity

## Preliminary Outputs

Representative intermediate outputs include:
- Unperturbed orbital motion and angular velocity
- Perturbed orbital motion and angular velocity

## Repository Structure
- 'src/' - Python implementation
- 'results/' - Key result figures (PNG)
- 'figures/' - Source figures (EPS)

## Reproducibility and external dependencies

This project relies on external SPICE kernels for ephemerides and constants, not included in this repository.
To run the simulations, the following NAIF kernels are required:
- 'naif0012.tls'
- 'de440s.bsp'
- 'gm_de440.tpc'
- 'pck00010.tpc'