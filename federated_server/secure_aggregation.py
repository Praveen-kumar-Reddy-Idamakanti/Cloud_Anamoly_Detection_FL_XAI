"""
Secure Aggregation Module for Federated Learning
Implements the secure aggregation protocol from "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
(Bonawitz et al., 2017)
"""
import os
import hashlib
import hmac
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Parameters for Diffie-Hellman key exchange
DH_PARAMETERS = dh.generate_parameters(generator=2, key_size=2048, backend=default_backend())

@dataclass
class ClientKeys:
    """Stores cryptographic keys for a client."""
    private_key: dh.DHPrivateKey
    public_key: dh.DHPublicKey
    shared_secrets: Dict[int, bytes]  # Maps client_id to shared secret

class SecureAggregator:
    """Implements secure aggregation for federated learning."""
    
    def __init__(self, num_clients: int, client_id: int):
        self.num_clients = num_clients
        self.client_id = client_id
        self.keys = self._generate_keys()
        self._setup_shared_secrets()
        
    def _generate_keys(self) -> ClientKeys:
        """Generate Diffie-Hellman key pair."""
        private_key = DH_PARAMETERS.generate_private_key()
        public_key = private_key.public_key()
        return ClientKeys(private_key, public_key, {})
    
    def _setup_shared_secrets(self):
        """Set up shared secrets with all other clients."""
        for other_id in range(self.num_clients):
            if other_id != self.client_id:
                # In a real implementation, this would involve key exchange with other clients
                # For now, we'll simulate it with a dummy secret
                self.keys.shared_secrets[other_id] = os.urandom(32)
    
    def mask_updates(self, parameters: List[np.ndarray], round_num: int) -> Tuple[List[np.ndarray], Dict]:
        """
        Apply secure masking to model parameters.
        
        Args:
            parameters: List of model parameters to mask
            round_num: Current round number for seed generation
            
        Returns:
            Tuple of (masked_parameters, masks_dict)
        """
        masked_parameters = [param.copy() for param in parameters]
        masks_dict = {}
        
        # Generate pairwise masks
        for other_id in range(self.num_clients):
            if other_id == self.client_id:
                continue
                
            # Generate a deterministic seed using the shared secret and round number
            seed = hmac.new(
                self.keys.shared_secrets[other_id], 
                str(round_num).encode(), 
                hashlib.sha256
            ).digest()
            
            # Generate random mask using the seed
            rng = np.random.RandomState(int.from_bytes(seed[:4], byteorder='big'))
            
            # Generate and apply masks for each parameter
            for i in range(len(parameters)):
                mask = rng.rand(*parameters[i].shape).astype(parameters[i].dtype)
                
                # For client i < j, add the mask; for i > j, subtract the mask
                if self.client_id < other_id:
                    masked_parameters[i] += mask
                else:
                    masked_parameters[i] -= mask
                
                # Store the mask for debugging/verification
                if other_id not in masks_dict:
                    masks_dict[other_id] = []
                masks_dict[other_id].append(mask)
        
        return masked_parameters, masks_dict
    
    @staticmethod
    def aggregate_updates(updates: List[Tuple[List[np.ndarray], int]], round_num: int) -> List[np.ndarray]:
        """
        Securely aggregate model updates from multiple clients.
        
        Args:
            updates: List of tuples containing (parameters, client_id) from each client
            round_num: Current round number
            
        Returns:
            Aggregated model parameters
        """
        if not updates:
            raise ValueError("No updates to aggregate")
            
        # Initialize with zeros of the same shape as the first update
        aggregated = [np.zeros_like(param) for param in updates[0][0]]
        
        # Sum all updates (masks will cancel out)
        for update, _ in updates:
            for i in range(len(update)):
                aggregated[i] += update[i] / len(updates)
                
        return aggregated

# Utility functions for encryption
def encrypt_parameters(parameters: List[np.ndarray], key: bytes) -> bytes:
    """Encrypt model parameters using AES."""
    # Serialize parameters
    serialized = b''.join(param.tobytes() for param in parameters)
    
    # Generate a random IV
    iv = os.urandom(16)
    
    # Pad the data
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(serialized) + padder.finalize()
    
    # Encrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    
    # Return IV + encrypted data
    return iv + encrypted

def decrypt_parameters(encrypted_data: bytes, key: bytes, param_shapes: List[tuple], param_dtypes: List[str]) -> List[np.ndarray]:
    """Decrypt model parameters."""
    # Extract IV and encrypted data
    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    
    # Decrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted) + decryptor.finalize()
    
    # Unpad
    unpadder = padding.PKCS7(128).unpadder()
    serialized = unpadder.update(padded_data) + unpadder.finalize()
    
    # Deserialize parameters
    parameters = []
    offset = 0
    for shape, dtype in zip(param_shapes, param_dtypes):
        size = np.prod(shape) * np.dtype(dtype).itemsize
        param_bytes = serialized[offset:offset+size]
        param = np.frombuffer(param_bytes, dtype=dtype).reshape(shape)
        parameters.append(param)
        offset += size
        
    return parameters
