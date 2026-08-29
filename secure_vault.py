import os
from cryptography.fernet import Fernet

class SecureVault:
    def __init__(self, key_path="secret.key"):
        """
        Initializes the cryptographic vault. 
        Automatically loads an existing key or generates a new one.
        """
        self.key_path = key_path
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        """Loads a master key from disk, or generates one if it doesn't exist."""
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        else:
            # Generate a secure random AES key
            new_key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(new_key)
            return new_key

    def encrypt_template(self, raw_data: bytes) -> bytes:
        """Encrypts raw biometric data (arrays/bytes) before writing to disk."""
        return self.cipher.encrypt(raw_data)

    def decrypt_template(self, encrypted_data: bytes) -> bytes:
        """Decrypts ciphertext from disk back into raw bytes for in-memory matching."""
        return self.cipher.decrypt(encrypted_data)