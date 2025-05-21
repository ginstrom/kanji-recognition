import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
import os
import sys # For manipulating sys.modules
# import json # Unused F401
import logging

# Attempt to import from src.improved_train
# This assumes that the tests are run from a context where 'src' is in PYTHONPATH
# or the current working directory is the project root.
try:
    from src.improved_train import (
        set_seed,
        ResidualBlock,
        ImprovedKanjiCNN,
        plot_training_history,
        train_model_improved
    )
except ImportError as e:
    # Fallback for cases where 'src' is not directly in path, common in some setups
    # This might happen if cwd is tests/
    # Adjust path to go up one level to project root, then into src
    # This is a common pattern but might need adjustment based on exact test runner setup
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from src.improved_train import (
            set_seed,
            ResidualBlock,
            ImprovedKanjiCNN,
            plot_training_history,
            train_model_improved
        )
    except ImportError:
        raise ImportError(f"Could not import from src.improved_train. Original error: {e}. Ensure PYTHONPATH is set or tests are run from project root.")


# Suppress logging for cleaner test output during tests
# Store original logging level
original_logging_level = logging.getLogger().getEffectiveLevel()

class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(original_logging_level) # Restore logging level

class TestSetSeed(BaseTestCase):
    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    @patch('src.improved_train.torch.backends.cudnn') # Ensure patching the one used by the module
    @patch('src.improved_train.logger') # Mock logger to check for info log
    def test_set_seed_execution(self, mock_logger, mock_cudnn, mock_cuda_manual_seed_all, mock_torch_manual_seed):
        with patch('torch.cuda.is_available', return_value=True):
            set_seed(42)
            mock_torch_manual_seed.assert_called_once_with(42)
            mock_cuda_manual_seed_all.assert_called_once_with(42)
            self.assertTrue(mock_cudnn.deterministic)
            self.assertFalse(mock_cudnn.benchmark)
            mock_logger.info.assert_called_with("Random seed set to 42")

        mock_torch_manual_seed.reset_mock()
        mock_cuda_manual_seed_all.reset_mock()
        mock_logger.reset_mock()

        with patch('torch.cuda.is_available', return_value=False):
            set_seed(100)
            mock_torch_manual_seed.assert_called_once_with(100)
            mock_cuda_manual_seed_all.assert_not_called()
            mock_logger.info.assert_called_with("Random seed set to 100")


class TestResidualBlock(BaseTestCase):
    def test_init(self):
        block_downsample_stride = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.assertIsInstance(block_downsample_stride.conv1, nn.Conv2d)
        self.assertEqual(block_downsample_stride.conv1.out_channels, 128)
        self.assertIsNotNone(block_downsample_stride.downsample)

        block_downsample_channels = ResidualBlock(in_channels=64, out_channels=128, stride=1)
        self.assertIsNotNone(block_downsample_channels.downsample)

        block_no_downsample = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        self.assertIsNone(block_no_downsample.downsample)

    def test_forward_pass_shapes(self):
        # No downsample
        block1 = ResidualBlock(in_channels=32, out_channels=32, stride=1)
        input_tensor1 = torch.randn(4, 32, 16, 16) # B, C, H, W
        output_tensor1 = block1(input_tensor1)
        self.assertEqual(output_tensor1.shape, (4, 32, 16, 16))

        # Downsample with stride
        block2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        input_tensor2 = torch.randn(4, 32, 16, 16)
        output_tensor2 = block2(input_tensor2)
        self.assertEqual(output_tensor2.shape, (4, 64, 8, 8))

        # Downsample with channels change only
        block3 = ResidualBlock(in_channels=32, out_channels=64, stride=1)
        input_tensor3 = torch.randn(4, 32, 16, 16)
        output_tensor3 = block3(input_tensor3)
        self.assertEqual(output_tensor3.shape, (4, 64, 16, 16))

class TestImprovedKanjiCNN(BaseTestCase):
    def test_init(self):
        model = ImprovedKanjiCNN(num_classes=100, dropout_rate=0.3)
        self.assertEqual(model.fc.out_features, 100)
        self.assertEqual(model.dropout.p, 0.3)
        self.assertIsInstance(model.initial_conv, nn.Conv2d)
        self.assertTrue(len(model.res_blocks) == 4) # As per current architecture

    @patch.object(ImprovedKanjiCNN, '_initialize_weights')
    def test_weight_initialization_called(self, mock_initialize_weights):
        ImprovedKanjiCNN(num_classes=10)
        mock_initialize_weights.assert_called_once()

    def test_forward_pass(self):
        model = ImprovedKanjiCNN(num_classes=50)
        # Test with a typical input size, e.g., 128x128 for Kanji
        input_tensor = torch.randn(2, 1, 128, 128) # B, C, H, W
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 50))

        # Test with another input size to check adaptability (due to AdaptiveAvgPool2d)
        input_tensor_small = torch.randn(3, 1, 64, 64)
        output_small = model(input_tensor_small)
        self.assertEqual(output_small.shape, (3, 50))

class TestPlotTrainingHistory(BaseTestCase):
    def setUp(self):
        self.history = {
            'train_loss': [0.5, 0.4], 'val_loss': [0.6, 0.5],
            'train_acc': [80.0, 85.0], 'val_acc': [75.0, 78.0],
            'best_val_acc': 78.0, 'best_epoch': 2
        }
        self.output_dir = "dummy_plot_output_dir"

    @patch('matplotlib.pyplot', create=True) # Added create=True
    @patch('os.makedirs')
    def test_plot_history_successful_run(self, mock_os_makedirs, mock_plt_module):
        plot_training_history(self.history, self.output_dir)

        fig_dir = os.path.join(self.output_dir, 'figures')
        mock_os_makedirs.assert_called_once_with(fig_dir, exist_ok=True)

        self.assertEqual(mock_plt_module.figure.call_count, 2)
        self.assertEqual(mock_plt_module.plot.call_count, 4)
        mock_plt_module.savefig.assert_any_call(os.path.join(fig_dir, 'loss_history.png'))
        mock_plt_module.savefig.assert_any_call(os.path.join(fig_dir, 'accuracy_history.png'))
        self.assertEqual(mock_plt_module.close.call_count, 2)

    @patch('src.improved_train.logger')
    def test_plot_history_matplotlib_not_available(self, mock_logger):
        original_pyplot = sys.modules.get('matplotlib.pyplot')
        sys.modules['matplotlib.pyplot'] = None # Simulate module not being importable
        
        # Ensure plot_training_history is re-evaluated with the modified sys.modules
        # This requires that the import statement inside plot_training_history fails.
        # No need to reload the entire 'src.improved_train' module if 'plot_training_history'
        # attempts its import ('import matplotlib.pyplot as plt') each time.
        # If 'plot_training_history' had 'plt' as a global in its module, reload would be needed.

        try:
            plot_training_history(self.history, self.output_dir)
            mock_logger.warning.assert_any_call("Matplotlib not available. Skipping history plots.")
        finally:
            # Restore matplotlib.pyplot in sys.modules
            if original_pyplot is not None:
                sys.modules['matplotlib.pyplot'] = original_pyplot
            elif 'matplotlib.pyplot' in sys.modules: # if it was None initially and got set
                del sys.modules['matplotlib.pyplot']


    @patch('matplotlib.pyplot', create=True) # Added create=True
    @patch('os.makedirs')
    @patch('src.improved_train.logger')
    def test_plot_history_savefig_fails(self, mock_logger, mock_os_makedirs, mock_plt_module):
        mock_plt_module.savefig.side_effect = Exception("Disk full")
        plot_training_history(self.history, self.output_dir)
        mock_logger.error.assert_any_call(unittest.mock.ANY) # Check that some error was logged
        # More specific check:
        self.assertTrue(any("Failed to save loss history plot" in call.args[0] for call in mock_logger.error.call_args_list))


class TestTrainModelImproved(BaseTestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=ImprovedKanjiCNN) # Use spec for better mocking
        self.mock_model.parameters.return_value = [nn.Parameter(torch.randn(1))] # For optimizer
        self.mock_model.to.return_value = self.mock_model
        # self.mock_model.return_value = torch.randn(2, 10) # Mock forward pass output B, NumClasses

        # Set up training attribute and train/eval methods for the mock model
        self.mock_model.training = False # Start in eval mode by default
        self.mock_model.train = MagicMock(side_effect=lambda: setattr(self.mock_model, 'training', True))
        self.mock_model.eval = MagicMock(side_effect=lambda: setattr(self.mock_model, 'training', False))

        self.mock_train_loader = MagicMock()
        self.mock_val_loader = MagicMock()
        self.mock_train_dataset = MagicMock()
        self.mock_val_dataset = MagicMock()

        self.mock_train_loader.dataset = self.mock_train_dataset
        self.mock_val_loader.dataset = self.mock_val_dataset
        
        # Define what iter(loader) returns using side_effect for new iterators
        def train_loader_iter_side_effect():
            # Returns an iterator for one epoch of training data
            return iter([
                (torch.randn(2, 1, 28, 28), torch.randint(0, 10, (2,))), # batch_size=2
            ])
        self.mock_train_loader.__iter__.side_effect = train_loader_iter_side_effect
        self.mock_train_loader.__len__.return_value = 1 # Number of batches per epoch

        # self.fixed_val_labels is defined later in setUp, use a placeholder for now if needed,
        # or ensure fixed_val_labels is defined before this.
        # Let's define fixed_val_labels earlier.
        self.fixed_val_outputs = torch.zeros(2, 10)  # batch_size=2, num_classes=10
        self.fixed_val_outputs[0, 1] = 1.0
        self.fixed_val_outputs[1, 0] = 1.0
        self.fixed_val_labels = torch.tensor([1, 0]) # batch_size=2

        def val_loader_iter_side_effect():
            # Returns an iterator for one epoch of validation data
            return iter([
                (torch.randn(2, 1, 28, 28), self.fixed_val_labels), # batch_size=2
            ])
        self.mock_val_loader.__iter__.side_effect = val_loader_iter_side_effect
        self.mock_val_loader.__len__.return_value = 1 # Number of batches per epoch

        self.device = torch.device('cpu')
        self.output_dir = "test_train_output_dir"
        self.num_epochs = 1
        
        # --- Fix for ensuring model save --- 
        # Define fixed outputs and labels for validation to ensure val_acc > 0
        # This is now done earlier in setUp to be available for val_loader_iter_side_effect
        # self.fixed_val_outputs = torch.zeros(2, 10) ...
        # self.fixed_val_labels = torch.tensor([1, 0])

        # self.mock_val_loader.__iter__.return_value = iter([ # This line is now handled by side_effect
        #     (torch.randn(2, 1, 28, 28), self.fixed_val_labels),
        # ])
        # self.mock_val_loader.__len__.return_value = 1 # Already set

        # Model side effect to control output based on training mode
        def model_side_effect(input_tensor):
            if not self.mock_model.training:  # Eval mode
                # Ensure the output batch size matches input tensor batch size
                return self.fixed_val_outputs[:input_tensor.size(0)]
            else:  # Train mode
                return torch.randn(input_tensor.size(0), 10) # num_classes = 10
        self.mock_model.side_effect = model_side_effect
        # --- End of fix for ensuring model save ---

        # Common patchers for train_model_improved tests
        self.patchers = {
            'torch_save': patch('src.improved_train.torch.save'),
            'json_dump': patch('src.improved_train.json.dump'),
            'open': patch('src.improved_train.open', new_callable=mock_open),
            'os_makedirs': patch('os.makedirs'),
            'plot_history': patch('src.improved_train.plot_training_history'),
            'tqdm': patch('src.improved_train.tqdm', lambda x, **kwargs: x), # Simple lambda to disable tqdm
            'logger': patch('src.improved_train.logger'),
            'grad_scaler': patch('src.improved_train.torch.cuda.amp.GradScaler')
        }
        self.mocks = {}

    def start_patchers(self):
        for name, p in self.patchers.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop) # Ensure patchers are stopped even if test fails

        # Mock GradScaler instance methods
        mock_scaler_instance = self.mocks['grad_scaler'].return_value
        mock_scaler_instance.scale.return_value = MagicMock(spec=torch.Tensor) # So it can be .backward()
        mock_scaler_instance.scale.return_value.backward = MagicMock()


    def tearDown(self):
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

    def test_label_to_idx_from_class_to_idx(self):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = {'a': 0, 'b': 1}
        self.mock_train_dataset.label_to_idx = None 
        self.mock_train_dataset.classes = None

        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=self.num_epochs, device=self.device, output_dir=self.output_dir)
        
        self.mocks['logger'].info.assert_any_call("Using 'class_to_idx' from dataset with 2 classes.")
        # Check checkpoint saving args for label_to_idx
        # Best model is saved first, then possibly intermediate.
        saved_checkpoint_data = self.mocks['torch_save'].call_args_list[0][0][0] # First arg of first call
        self.assertEqual(saved_checkpoint_data['label_to_idx'], {'a': 0, 'b': 1})

    def test_label_to_idx_from_label_to_idx_attr(self):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = None
        self.mock_train_dataset.label_to_idx = {'x': 0, 'y': 1}
        self.mock_train_dataset.classes = None

        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=self.num_epochs, device=self.device, output_dir=self.output_dir)
        self.mocks['logger'].info.assert_any_call("Using 'label_to_idx' from dataset with 2 classes.")
        saved_checkpoint_data = self.mocks['torch_save'].call_args_list[0][0][0]
        self.assertEqual(saved_checkpoint_data['label_to_idx'], {'x': 0, 'y': 1})

    def test_label_to_idx_from_classes_attr(self):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = None
        self.mock_train_dataset.label_to_idx = None
        self.mock_train_dataset.classes = ['cat', 'dog']

        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=self.num_epochs, device=self.device, output_dir=self.output_dir)
        self.mocks['logger'].info.assert_any_call("Constructed label_to_idx from dataset.classes with 2 classes.")
        saved_checkpoint_data = self.mocks['torch_save'].call_args_list[0][0][0]
        self.assertEqual(saved_checkpoint_data['label_to_idx'], {'cat': 0, 'dog': 1})
    
    def test_label_to_idx_warning_if_missing(self):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = None
        self.mock_train_dataset.label_to_idx = None
        self.mock_train_dataset.classes = None
        
        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=self.num_epochs, device=self.device, output_dir=self.output_dir)
        self.mocks['logger'].warning.assert_any_call(unittest.mock.ANY)
        self.assertTrue(any("Could not find 'class_to_idx'" in call.args[0] for call in self.mocks['logger'].warning.call_args_list))

    def test_training_loop_basic_flow_cpu(self):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = {'a':0} # Minimal label_to_idx

        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=self.num_epochs, device=self.device, output_dir=self.output_dir)

        self.mock_model.train.assert_called()
        self.mock_model.eval.assert_called()
        self.mock_model.to.assert_called_with(self.device)
        
        # Check optimizer steps (mocked through scaler)
        scaler_instance = self.mocks['grad_scaler'].return_value
        scaler_instance.scale.assert_called() # Called with loss
        scaler_instance.scale.return_value.backward.assert_called() # loss.backward()
        scaler_instance.step.assert_called()
        scaler_instance.update.assert_called()

        self.mocks['torch_save'].assert_called() # For best model
        history_file_path = os.path.join(self.output_dir, 'training_history.json')
        self.mocks['open'].assert_any_call(history_file_path, 'w')
        self.mocks['json_dump'].assert_called()
        self.mocks['plot_history'].assert_called_once()
        self.mocks['os_makedirs'].assert_any_call(self.output_dir, exist_ok=True)

    @patch('torch.cuda.is_available', return_value=True)
    def test_training_loop_cuda_device(self, mock_cuda_is_available):
        self.start_patchers()
        cuda_device = torch.device('cuda')
        self.mock_train_dataset.class_to_idx = {'a':0}

        original_tensor_to = torch.Tensor.to
        def mock_tensor_to(self_tensor, *args, **kwargs):
            # If trying to move to CUDA, just return self (noop)
            if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
                return self_tensor
            if 'device' in kwargs and isinstance(kwargs['device'], torch.device) and kwargs['device'].type == 'cuda':
                return self_tensor
            # Call the original method for other cases (e.g., .to(cpu_device) or .to(dtype))
            return original_tensor_to(self_tensor, *args, **kwargs)

        with patch('torch.Tensor.to', new=mock_tensor_to):
            train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                                 num_epochs=self.num_epochs, device=cuda_device, output_dir=self.output_dir)
        
        self.mock_model.to.assert_called_with(cuda_device)
        self.mocks['grad_scaler'].assert_called_with(enabled=True)
        self.mocks['logger'].info.assert_any_call("Cleared CUDA cache after training.")


    @patch('src.improved_train.torch.load')
    def test_early_stopping(self, mock_torch_load):
        self.start_patchers()
        self.mock_train_dataset.class_to_idx = {'a': 0}
        
        # Simulate validation accuracy not improving for `patience` epochs
        # To do this, we need the val_loader to yield different results or mock criterion/model output
        # Simpler: mock the part that calculates val_acc
        
        # For early stopping, we need more than 1 epoch.
        # We also need to control the val_acc.
        # The `train_model_improved` has `patience=5` by default.
        
        # This test is more complex due to the internal epoch loop.
        # A simpler check is if the log message for patience appears.
        
        # Let's make val_acc always 0.5, so it never improves after the first epoch (where best_val_acc is 0)
        fixed_val_outputs_for_test = torch.tensor([[1.0, 0.0]] * 2) # Mock model output for val, batch_size=2
        fixed_val_labels_for_test = torch.tensor([0] * 2)         # Mock labels for val, batch_size=2
    
        original_model_call = self.mock_model.side_effect # Store original if any
    
        def val_side_effect_for_test(*args, **kwargs): # Renamed to avoid conflict
            input_tensor = args[0] if args else kwargs.get('input')
            if self.mock_model.training: # If model.train() was called
                if original_model_call: # This is model_side_effect from setUp
                    return original_model_call(input_tensor) 
                # Fallback if original_model_call was None, ensure correct batch size
                return torch.randn(input_tensor.size(0), fixed_val_outputs_for_test.size(1)) # Use num_classes from test's fixed_val_outputs
            return fixed_val_outputs_for_test[:input_tensor.size(0)] # Fixed output for validation, match batch size
    
        self.mock_model.side_effect = val_side_effect_for_test # Output for val to make acc constant
        
        # Configure val_loader specifically for this test
        def early_stopping_val_loader_iter_side_effect():
            return iter([
                (torch.randn(2, 1, 28, 28), fixed_val_labels_for_test), # Use labels specific to this test
            ])
        self.mock_val_loader.__iter__.side_effect = early_stopping_val_loader_iter_side_effect
        self.mock_val_loader.__len__.return_value = 1 # One batch for validation
    
    
        # Mock torch.load to return something minimal for the "best model loading" at the end
        mock_torch_load.return_value = {
            'epoch': 1, 
            'model_state_dict': self.mock_model.state_dict(), # Mock
            'val_acc': 0.5 
        }
        
        # Run for enough epochs to trigger early stopping (patience=5)
        # It should stop after 1 (initial best) + 5 (patience_counter) = 6 epochs
        train_model_improved(self.mock_model, self.mock_train_loader, self.mock_val_loader,
                             num_epochs=10, device=self.device, output_dir=self.output_dir)

        self.mocks['logger'].info.assert_any_call("Early stopping triggered after 6 epochs")
        # Check that only 6 epochs ran (logged by tqdm or epoch logging)
        # Example: check logger for "Epoch [6/10]" but not "Epoch [7/10]"
        # This specific check depends on exact logging format.
        # A more robust check is that torch.save for best model was called for epoch 1
        # and then no more "Saved best model" logs for a while.
        
        # Check that best model (from epoch 1) was saved
        first_save_call = self.mocks['torch_save'].call_args_list[0][0][0]
        self.assertEqual(first_save_call['epoch'], 1)
        self.assertAlmostEqual(first_save_call['val_acc'], 100.0) # val_acc is 1.0 (100%)

if __name__ == '__main__':
    unittest.main() 