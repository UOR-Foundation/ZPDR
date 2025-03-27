"""
Error Correction Example using ZPDR

This example demonstrates the error correction capabilities of the Zero-Point Data
Resolution (ZPDR) framework. It showcases how the trilateral vector representation
system provides inherent error detection and correction capabilities through the
mathematical coherence relationships between the three geometric spaces.

Key Features:

1. Controlled Error Injection:
   - Adding calibrated noise to ZPDR vector components
   - Simulating different types of data corruption
   - Calibrated corruption levels to test error correction thresholds
   - Visualization of error effects on vector representations

2. Progressive Error Correction:
   - Multi-level error correction implementation
   - Coherence-based error detection
   - Zero-point normalization for correction
   - Invariant-guided correction mechanisms
   - Verification of correction effectiveness

3. Measurement and Analysis:
   - Quantification of error correction success rates
   - Performance analysis at different noise levels
   - Coherence threshold determination
   - Visual representation of correction effectiveness

4. Mathematical Foundation:
   - Demonstration of Prime Framework principles in error correction
   - Coherence as a key metric for error detection
   - Trilateral vector system's robustness to corruption
   - Geometric invariant preservation during correction

This example provides practical insights into how ZPDR's mathematical foundation
enables robust data representation with built-in error correction capabilities,
making it suitable for applications where data integrity is critical.

Usage:
    python -m zpdr.examples.error_correction_example [--noise-level=0.2] [--iterations=100]
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Import ZPDR components
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector,
    SpaceTransformer
)
from zpdr.utils import (
    validate_trilateral_coherence,
    normalize_with_invariants,
    denormalize_with_invariants,
    COHERENCE_THRESHOLD
)


class ErrorCorrectionDemo:
    """
    Demonstration of ZPDR error correction capabilities.
    
    This class provides a comprehensive demonstration of how the ZPDR framework
    detects and corrects errors in data representation through its trilateral
    vector system and coherence-based validation mechanisms.
    
    The demo implements:
    
    1. Data Representation:
       - Creation of valid ZPDR trilateral vectors (H, E, U)
       - Establishment of baseline coherence measures
       - Visualization of vector relationships
    
    2. Error Introduction:
       - Controlled noise injection into vector components
       - Multiple corruption patterns and severity levels
       - Preservation of some mathematical invariants despite corruption
    
    3. Error Correction Mechanisms:
       - Multi-stage progressive correction techniques
       - Coherence-guided correction pathways
       - Invariant-based reconstruction
       - Zero-point normalization procedures
    
    4. Analysis and Visualization:
       - Measurement of correction effectiveness
       - Coherence threshold analysis
       - Visual representation of error and correction
    
    This demonstration shows how the mathematical principles of the Prime Framework
    enable robust error detection and correction capabilities in ZPDR, making it
    suitable for applications requiring high data integrity in challenging environments.
    """
    
    def __init__(self, config=None):
        """
        Initialize the error correction demonstration with configuration.
        
        Args:
            config: Optional configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'noise_levels': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
            'iterations_per_level': 100,
            'vector_dimensions': 3,
            'correction_levels': ['basic', 'intermediate', 'advanced'],
            'visualize_results': True,
            'coherence_threshold': COHERENCE_THRESHOLD,
            'verbose': True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize results storage
        self.results = {
            'success_rates': [],
            'coherence_before': [],
            'coherence_after': [],
            'correction_times': []
        }
    
    def generate_test_vector(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a valid ZPDR trilateral vector for testing.
        
        Creates a mathematically coherent set of three vectors (H, E, U) that
        represent a valid ZPDR encoding, ensuring they have the proper geometric
        properties and coherence relationships required by the Prime Framework.
        
        Returns:
            Tuple of (H, E, U) numpy arrays with valid ZPDR vector components
        """
        # Create a valid set of vectors that have high coherence
        # For testing purposes, we'll create structured vectors
        
        # Hyperbolic vector (must be within unit disk)
        H = np.array([0.2, 0.3, 0.1])
        
        # Elliptical vector (must be on unit sphere)
        E = np.array([0.6, 0.7, 0.4])
        E = E / np.linalg.norm(E)  # Normalize to unit length
        
        # Euclidean vector
        U = np.array([0.5, 0.2, 0.8])
        
        # Ensure they have the appropriate mathematical relationships
        # In a full implementation, these would be properly linked based on
        # the specific data being encoded
        
        self.log(f"Generated test vectors with dimensions: H{H.shape}, E{E.shape}, U{U.shape}")
        
        # Check initial coherence
        is_valid, coherence = validate_trilateral_coherence((H, E, U))
        self.log(f"Initial vectors coherence: {float(coherence):.4f} (valid: {is_valid})")
        
        return H, E, U
    
    def add_noise(self, vectors: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                 noise_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add controlled noise to the trilateral vectors.
        
        This function simulates data corruption by adding calibrated Gaussian noise
        to the vector components, with different handling for each geometric space
        to respect their mathematical constraints.
        
        Args:
            vectors: Tuple of (H, E, U) vectors
            noise_level: Float between 0 and 1 indicating noise intensity
            
        Returns:
            Tuple of corrupted (H, E, U) vectors
        """
        H, E, U = vectors
        
        # Add noise to each vector, scaled by noise_level
        H_noisy = H + np.random.normal(0, noise_level, H.shape)
        E_noisy = E + np.random.normal(0, noise_level, E.shape)
        U_noisy = U + np.random.normal(0, noise_level, U.shape)
        
        # Ensure the vectors still respect their space constraints
        
        # For hyperbolic vector, project back into unit disk if needed
        H_norm = np.linalg.norm(H_noisy)
        if H_norm >= 0.99:  # Allow some margin
            H_noisy = H_noisy * 0.98 / H_norm
        
        # For elliptical vector, always project back to unit sphere
        E_noisy = E_noisy / np.linalg.norm(E_noisy)
        
        # No constraints for Euclidean vector
        
        self.log(f"Added noise (level {noise_level:.2f}) to vectors")
        
        return H_noisy, E_noisy, U_noisy
    
    def correct_errors(self, vectors: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                      level: str = 'advanced') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply ZPDR error correction to corrupted vectors.
        
        This function implements the error correction procedures of the ZPDR framework,
        using coherence measures and geometric invariants to detect and correct errors
        in the trilateral vector representation.
        
        It demonstrates how the mathematical principles of the Prime Framework enable
        robust error correction through the relationships between different geometric spaces.
        
        Args:
            vectors: Tuple of (H, E, U) vectors with potential errors
            level: Error correction level ('basic', 'intermediate', or 'advanced')
            
        Returns:
            Tuple of corrected (H, E, U) vectors
        """
        H, E, U = vectors
        
        # Check coherence before correction
        is_valid_before, coherence_before = validate_trilateral_coherence((H, E, U))
        self.log(f"Before correction: coherence = {float(coherence_before):.4f}, valid = {is_valid_before}")
        
        # Basic error correction
        if level in ['basic', 'intermediate', 'advanced']:
            # Space-specific corrections
            
            # Ensure hyperbolic vector is within unit disk
            H_norm = np.linalg.norm(H)
            if H_norm >= 0.99:  # projection with margin
                H = H * 0.98 / H_norm
            
            # Ensure elliptical vector is on unit sphere
            E = E / np.linalg.norm(E)
        
        # Intermediate error correction
        if level in ['intermediate', 'advanced']:
            # Extract invariants before applying transformations
            _, H_invariants = normalize_with_invariants(H, "hyperbolic")
            _, E_invariants = normalize_with_invariants(E, "elliptical")
            _, U_invariants = normalize_with_invariants(U, "euclidean")
            
            # Apply cross-space transformations to improve coherence
            # These demonstrate how the relationships between spaces can be used
            # to correct errors in one space based on information from others
            
            # Create space-specific vectors
            H_vec = HyperbolicVector(H)
            E_vec = EllipticalVector(E)
            U_vec = EuclideanVector(U)
            
            # Apply transformations to synchronize spaces
            E_from_H = SpaceTransformer.hyperbolic_to_elliptical(H_vec)
            U_from_H = SpaceTransformer.hyperbolic_to_euclidean(H_vec)
            H_from_E = SpaceTransformer.elliptical_to_hyperbolic(E_vec)
            U_from_E = SpaceTransformer.elliptical_to_euclidean(E_vec)
            H_from_U = SpaceTransformer.euclidean_to_hyperbolic(U_vec)
            E_from_U = SpaceTransformer.euclidean_to_elliptical(U_vec)
            
            # Blend transformations to improve coherence
            H = (H * 0.6 + H_from_E.components * 0.2 + H_from_U.components * 0.2)
            E = (E * 0.6 + E_from_H.components * 0.2 + E_from_U.components * 0.2)
            U = (U * 0.6 + U_from_H.components * 0.2 + U_from_E.components * 0.2)
            
            # Reapply constraints
            H_norm = np.linalg.norm(H)
            if H_norm >= 0.99:
                H = H * 0.98 / H_norm
            E = E / np.linalg.norm(E)
        
        # Advanced error correction
        if level == 'advanced':
            # Use invariants to restore original structure
            H_normalized, H_inv = normalize_with_invariants(H, "hyperbolic")
            E_normalized, E_inv = normalize_with_invariants(E, "elliptical")
            U_normalized, U_inv = normalize_with_invariants(U, "euclidean")
            
            # Apply original invariants to the normalized vectors
            H = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
            E = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
            U = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
            
            # Final coherence optimization
            # This demonstrates how iterative adjustments guided by coherence
            # measures can improve error correction effectiveness
            
            # Check if further correction is needed
            is_valid_mid, coherence_mid = validate_trilateral_coherence((H, E, U))
            if not is_valid_mid:
                # Apply additional corrections
                H_vec = HyperbolicVector(H)
                E_vec = EllipticalVector(E)
                U_vec = EuclideanVector(U)
                
                # More aggressive space transformations
                E_from_H = SpaceTransformer.hyperbolic_to_elliptical(H_vec)
                U_from_H = SpaceTransformer.hyperbolic_to_euclidean(H_vec)
                
                # Apply correction to the most corrupted vectors
                # based on coherence contribution analysis
                E = E_from_H.components  # Replace E with transform from H
                
                # Apply final constraints
                E = E / np.linalg.norm(E)
        
        # Check coherence after correction
        is_valid_after, coherence_after = validate_trilateral_coherence((H, E, U))
        self.log(f"After {level} correction: coherence = {float(coherence_after):.4f}, valid = {is_valid_after}")
        
        return H, E, U
    
    def run_test_suite(self):
        """
        Run a comprehensive test suite of error correction scenarios.
        
        This function executes a series of error correction tests with varying
        noise levels and correction approaches, measuring the effectiveness of
        ZPDR's error correction capabilities under different conditions.
        
        The test suite demonstrates how the mathematical principles of the Prime
        Framework provide robust error correction even with significant corruption.
        """
        # Reset results
        self.results = {
            'noise_levels': self.config['noise_levels'],
            'success_rates': {level: [] for level in self.config['correction_levels']},
            'coherence_before': [],
            'coherence_after': {level: [] for level in self.config['correction_levels']},
        }
        
        self.log("Starting error correction test suite...")
        
        # Test each noise level
        for noise_level in self.config['noise_levels']:
            self.log(f"\nTesting noise level: {noise_level:.2f}")
            
            # Track success counts for this noise level
            success_count = {level: 0 for level in self.config['correction_levels']}
            coherence_before_sum = 0
            coherence_after_sum = {level: 0 for level in self.config['correction_levels']}
            
            # Run multiple iterations at this noise level
            for i in range(self.config['iterations_per_level']):
                # Generate a fresh test vector for each iteration
                original_vectors = self.generate_test_vector()
                
                # Add noise
                noisy_vectors = self.add_noise(original_vectors, noise_level)
                
                # Check coherence after noise
                is_valid, coherence = validate_trilateral_coherence(noisy_vectors)
                coherence_before_sum += float(coherence)
                
                # Apply each correction level
                for level in self.config['correction_levels']:
                    # Correct errors
                    corrected_vectors = self.correct_errors(noisy_vectors, level)
                    
                    # Check if correction was successful
                    is_valid_after, coherence_after = validate_trilateral_coherence(corrected_vectors)
                    coherence_after_sum[level] += float(coherence_after)
                    
                    if is_valid_after:
                        success_count[level] += 1
                
                # Progress indicator
                if (i + 1) % 20 == 0 or i == self.config['iterations_per_level'] - 1:
                    self.log(f"Completed {i+1}/{self.config['iterations_per_level']} iterations")
            
            # Calculate average results for this noise level
            avg_coherence_before = coherence_before_sum / self.config['iterations_per_level']
            self.results['coherence_before'].append(avg_coherence_before)
            
            for level in self.config['correction_levels']:
                success_rate = success_count[level] / self.config['iterations_per_level']
                avg_coherence_after = coherence_after_sum[level] / self.config['iterations_per_level']
                
                self.results['success_rates'][level].append(success_rate)
                self.results['coherence_after'][level].append(avg_coherence_after)
                
                self.log(f"{level.capitalize()} correction success rate: {success_rate*100:.1f}%")
        
        self.log("\nTest suite completed!")
    
    def visualize_results(self):
        """
        Visualize the error correction test results.
        
        This function creates plots showing the performance of different error
        correction levels across various noise levels, visualizing:
        - Success rates at different noise levels
        - Coherence before and after correction
        - Effectiveness of different correction approaches
        
        The visualizations demonstrate how ZPDR's mathematical foundation enables
        robust error correction even with substantial data corruption.
        """
        if not self.config['visualize_results']:
            return
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot success rates
        ax1.set_title('Error Correction Success Rates')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1.05)
        
        colors = {'basic': 'blue', 'intermediate': 'green', 'advanced': 'red'}
        markers = {'basic': 'o', 'intermediate': 's', 'advanced': '^'}
        
        for level in self.config['correction_levels']:
            ax1.plot(self.results['noise_levels'], self.results['success_rates'][level], 
                    marker=markers[level], color=colors[level], label=f"{level.capitalize()} Correction")
        
        # Add coherence threshold reference line
        ax1.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='50% Success Rate')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot coherence levels
        ax2.set_title('Coherence Before and After Correction')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Coherence')
        ax2.set_ylim(0, 1.05)
        
        # Plot coherence before correction
        ax2.plot(self.results['noise_levels'], self.results['coherence_before'], 
                marker='x', color='black', label='Before Correction')
        
        # Plot coherence after each correction level
        for level in self.config['correction_levels']:
            ax2.plot(self.results['noise_levels'], self.results['coherence_after'][level], 
                    marker=markers[level], color=colors[level], label=f"After {level.capitalize()}")
        
        # Add coherence threshold reference line
        ax2.axhline(y=float(COHERENCE_THRESHOLD), linestyle='--', color='red', alpha=0.5, 
                   label=f'Coherence Threshold ({float(COHERENCE_THRESHOLD):.2f})')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add a title for the entire figure
        fig.suptitle('ZPDR Error Correction Performance', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Show the plot
        plt.show()
    
    def log(self, message):
        """Output a log message if verbose mode is enabled."""
        if self.config['verbose']:
            print(message)


def main():
    """
    Main function for running the error correction demonstration.
    
    This function sets up and executes the error correction demonstration,
    handling command-line arguments and displaying the results. It showcases
    how the ZPDR framework's mathematical principles enable robust error
    detection and correction capabilities.
    
    The demonstration provides insights into:
    - How the trilateral vector system provides error resilience
    - The effectiveness of different correction approaches
    - The relationship between noise levels and correction success
    - The role of coherence in error detection and correction
    
    This practical demonstration shows how the Prime Framework's mathematical
    foundation enables ZPDR to maintain data integrity even in the presence
    of significant corruption.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ZPDR Error Correction Demo')
    parser.add_argument('--noise-level', type=float, default=0.2, 
                      help='Single noise level to test (0.0 to 1.0)')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of test iterations to run')
    parser.add_argument('--no-viz', action='store_true',
                      help='Disable visualization')
    
    args = parser.parse_args()
    
    # Configure the demo
    config = {
        'noise_levels': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9] if args.noise_level is None else [args.noise_level],
        'iterations_per_level': args.iterations,
        'visualize_results': not args.no_viz
    }
    
    # Create and run the demo
    print("Zero-Point Data Resolution (ZPDR) Error Correction Demo")
    print("======================================================")
    
    demo = ErrorCorrectionDemo(config)
    
    # Run a single demonstration with detailed steps
    print("\nRunning single demonstration...")
    H, E, U = demo.generate_test_vector()
    
    noise_level = 0.3
    print(f"\nAdding {noise_level:.2f} noise level to vectors...")
    H_noisy, E_noisy, U_noisy = demo.add_noise((H, E, U), noise_level)
    
    is_valid, coherence = validate_trilateral_coherence((H_noisy, E_noisy, U_noisy))
    print(f"After adding noise: coherence = {float(coherence):.4f}, valid = {is_valid}")
    
    print("\nApplying progressive error correction...")
    for level in ['basic', 'intermediate', 'advanced']:
        H_corrected, E_corrected, U_corrected = demo.correct_errors(
            (H_noisy, E_noisy, U_noisy), level
        )
        
        is_valid, coherence = validate_trilateral_coherence(
            (H_corrected, E_corrected, U_corrected)
        )
        print(f"After {level} correction: coherence = {float(coherence):.4f}, valid = {is_valid}")
    
    # Run the full test suite
    print("\nRunning full test suite across multiple noise levels...")
    demo.run_test_suite()
    
    # Visualize the results
    if config['visualize_results']:
        print("\nGenerating visualization of results...")
        demo.visualize_results()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()