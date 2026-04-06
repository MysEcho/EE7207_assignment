import torch
import unittest
import warnings
from models import FinBERTLoRAModel 

class TestFinBERTLoRAModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        setUpClass runs once before any tests. Innitialize model 
        and creates dummy data tensors.
        """

        warnings.filterwarnings("ignore")
        
        cls.model = FinBERTLoRAModel(
            model_name="hf-internal-testing/tiny-random-bert", 
            num_labels=3, 
            learning_rate=2e-4
        )
        
        # Dummy batch dimensions
        cls.batch_size = 4
        cls.seq_len = 16
        
        # Dummy tensors 
        cls.dummy_input_ids = torch.randint(0, 1000, (cls.batch_size, cls.seq_len))
        cls.dummy_attention_mask = torch.ones((cls.batch_size, cls.seq_len), dtype=torch.long)
        cls.dummy_labels = torch.randint(0, 3, (cls.batch_size,), dtype=torch.long)
        
        cls.dummy_batch = {
            'input_ids': cls.dummy_input_ids,
            'attention_mask': cls.dummy_attention_mask,
            'labels': cls.dummy_labels
        }

    def test_model_initialization(self):
        """Test if the model initializes correctly and applies LoRA."""

        # Check if the LoRA wrapper was applied successfully
        self.assertTrue(hasattr(self.model, 'model'))
        
        # Ensure the classifier head has the correct number of outputs
        self.assertEqual(self.model.hparams.num_labels, 3)

    def test_forward_pass_shape(self):
        """Test if the forward pass accepts inputs and outputs in the correct logit shape."""
        logits = self.model(self.dummy_input_ids, self.dummy_attention_mask)
        
        # [batch_size, num_classes] -> [4, 3]
        self.assertEqual(logits.shape, (self.batch_size, 3))

    def test_training_step(self):
        """Test if the training step successfully computes a valid loss."""

        loss = self.model.training_step(self.dummy_batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Check if Loss is not NaN or inf
        self.assertFalse(torch.isnan(loss).item(), "Training loss is NaN")
        self.assertFalse(torch.isinf(loss).item(), "Training loss is Infinite")

    def test_validation_step(self):
        """Test if the validation step executes without crashing and returns a valid loss."""
        loss = self.model.validation_step(self.dummy_batch, batch_idx=0)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).item(), "Validation loss is NaN")

    def test_optimizer_configuration(self):
        """Ensure the AdamW optimizer is correctly configured."""
        optimizer = self.model.configure_optimizers()
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        
        self.assertEqual(optimizer.param_groups[0]['lr'], 2e-4)

if __name__ == '__main__':

    unittest.main(verbosity=2)