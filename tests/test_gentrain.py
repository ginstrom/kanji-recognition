import unittest
from unittest.mock import patch, MagicMock, call # mock_open, os unused F401
# import os # Unused F401
import lmdb
import json
import pickle
from collections import defaultdict

# Use specific imports similar to test_genkanji.py
from src.gentrain import (
    # parse_args, # Unused F401
    _read_lmdb_metadata_and_info,
    _calculate_target_map_size,
    _transfer_data_and_update_stats,
    _update_train_db_metadata,
    merge_font_data_to_train
)

# Helper to create a dummy LMDB environment for mocking
def create_mock_lmdb_env(db_data=None, metadata=None, info=None):
    env = MagicMock(spec=lmdb.Environment)
    txn = MagicMock(spec=lmdb.Transaction)
    cursor = MagicMock(spec=lmdb.Cursor)

    env.begin.return_value.__enter__.return_value = txn
    txn.cursor.return_value = cursor
    
    # stat for counting entries
    txn.stat.return_value = {'entries': len(db_data) + 1 if db_data else 1} # +1 for metadata

    # Mocking get for metadata
    if metadata:
        txn.get.side_effect = lambda key: json.dumps(metadata).encode('utf-8') if key == b'__metadata__' else None
    else:
        txn.get.return_value = None

    # Mocking iteration for cursor
    if db_data:
        # Ensure db_data items are (key_bytes, value_bytes)
        cursor.__iter__.return_value = iter([(k.encode(), v) for k, v in db_data.items()])
    else:
        cursor.__iter__.return_value = iter([])

    # Mocking put for writing data
    txn.put = MagicMock()

    # Mocking env.info()
    default_info = {'map_size': 100*1024*1024, 'map_used': 0, 'last_pgno': 0, 'last_txnid': 0, 'max_readers': 126, 'num_readers': 0}
    env.info.return_value = {**default_info, **(info or {})}
    
    return env, txn, cursor

class TestGentrainHelpers(unittest.TestCase):

    @patch('os.path.exists')
    @patch('lmdb.open')
    def test_read_lmdb_metadata_and_info_success(self, mock_lmdb_open, mock_os_exists):
        mock_os_exists.return_value = True
        mock_meta = {"num_samples": 10, "key": "value"}
        mock_info_data = {'map_size': 200000000, 'map_used': 5000000}
        mock_env, _, _ = create_mock_lmdb_env(metadata=mock_meta, info=mock_info_data)
        mock_lmdb_open.return_value = mock_env

        meta, info, num_samples, map_size, map_used = _read_lmdb_metadata_and_info("dummy_path", "TestDB")

        self.assertEqual(meta, mock_meta)
        self.assertIn('map_size', info) # Check if info is populated
        self.assertEqual(num_samples, 10)
        self.assertEqual(map_size, mock_info_data['map_size'])
        self.assertEqual(map_used, mock_info_data['map_used'])
        mock_lmdb_open.assert_called_once_with("dummy_path", readonly=True, lock=False, readahead=False, meminit=False)

    @patch('os.path.exists', return_value=False)
    def test_read_lmdb_metadata_and_info_db_not_exist(self, mock_os_exists):
        meta, info, num_samples, map_size, map_used = _read_lmdb_metadata_and_info("non_existent_path", "TestDB")
        self.assertEqual(meta, {})
        self.assertEqual(info, {})
        self.assertEqual(num_samples, 0)
        self.assertTrue(map_size > 0) # Default map size
        self.assertEqual(map_used, 0)

    def test_calculate_target_map_size(self):
        min_map_size_constant = 10 * 1024 * 1024 * 1024  # 10GB

        # Scenario 1 (Basic):
        current_map_size_s1 = 1000
        current_data_size_s1 = 500
        num_current_items_s1 = 10
        num_new_items_s1 = 20
        # Note: The function _calculate_target_map_size takes an optional 5th arg 'override_map_size_gb'
        # which is not passed in these test scenarios, so it defaults to None in the function.
        size_s1 = _calculate_target_map_size(current_map_size_s1, current_data_size_s1, num_current_items_s1, num_new_items_s1)
        avg_item_size_s1 = current_data_size_s1 / num_current_items_s1 if num_current_items_s1 > 0 else (20 * 1024)
        additional_data_needed_s1 = num_new_items_s1 * avg_item_size_s1
        calculated_target_map_size_s1 = int(current_data_size_s1 + additional_data_needed_s1 * 10.0)
        expected_s1 = max(max(calculated_target_map_size_s1, current_map_size_s1), min_map_size_constant)
        self.assertEqual(size_s1, expected_s1)

        # Scenario 2 (Empty train DB):
        current_map_size_s2 = 1024*1024*100 # 100MB
        current_data_size_s2 = 0
        num_current_items_s2 = 0
        num_new_items_s2 = 100
        size_s2 = _calculate_target_map_size(current_map_size_s2, current_data_size_s2, num_current_items_s2, num_new_items_s2)
        avg_item_size_s2 = 20 * 1024 # default_item_size from src
        additional_data_needed_s2 = num_new_items_s2 * avg_item_size_s2
        calculated_target_map_size_s2 = int(current_data_size_s2 + additional_data_needed_s2 * 10.0)
        expected_s2 = max(max(calculated_target_map_size_s2, current_map_size_s2), min_map_size_constant)
        self.assertEqual(size_s2, expected_s2)
        
        # Scenario 3 (Calculated size less than current map_size):
        current_map_size_s3 = 200*1024*1024 # 200MB
        current_data_size_s3 = 50*1024*1024 # 50MB
        num_current_items_s3 = 1000
        num_new_items_s3 = 10
        size_s3 = _calculate_target_map_size(current_map_size_s3, current_data_size_s3, num_current_items_s3, num_new_items_s3)
        avg_item_size_s3 = current_data_size_s3 / num_current_items_s3 if num_current_items_s3 > 0 else (20 * 1024)
        additional_data_needed_s3 = num_new_items_s3 * avg_item_size_s3
        calculated_target_map_size_s3 = int(current_data_size_s3 + additional_data_needed_s3 * 10.0)
        expected_s3 = max(max(calculated_target_map_size_s3, current_map_size_s3), min_map_size_constant)
        self.assertEqual(size_s3, expected_s3)

        # Scenario 4 (Calculated size less than min_map_size (10GB), current also less):
        current_map_size_s4 = 10*1024*1024 # 10MB
        current_data_size_s4 = 5*1024*1024  # 5MB
        num_current_items_s4 = 100
        num_new_items_s4 = 1
        size_s4 = _calculate_target_map_size(current_map_size_s4, current_data_size_s4, num_current_items_s4, num_new_items_s4)
        avg_item_size_s4 = current_data_size_s4 / num_current_items_s4 if num_current_items_s4 > 0 else (20*1024)
        additional_data_needed_s4 = num_new_items_s4 * avg_item_size_s4
        calculated_target_map_size_s4 = int(current_data_size_s4 + additional_data_needed_s4 * 10.0)
        expected_s4 = max(max(calculated_target_map_size_s4, current_map_size_s4), min_map_size_constant)
        self.assertEqual(size_s4, expected_s4)

    def test_transfer_data_and_update_stats(self):
        mock_font_data = {
            "charA_font1": b"imgA1_data",
            "charB_font1": b"imgB1_data",
            "charA_font2": b"imgA2_data",
        }
        mock_font_env, _, _ = create_mock_lmdb_env(db_data=mock_font_data)
        mock_train_env, mock_train_txn, _ = create_mock_lmdb_env()

        current_train_samples = 5
        initial_train_counts = defaultdict(int, {'charC': 3})

        num_added, added_summary, final_counts, next_idx = _transfer_data_and_update_stats(
            mock_font_env, mock_train_env, current_train_samples, initial_train_counts, len(mock_font_data)
        )

        self.assertEqual(num_added, 3)
        self.assertEqual(added_summary, {'charA': 2, 'charB': 1})
        self.assertEqual(final_counts, {'charA': 2, 'charB': 1, 'charC': 3})
        self.assertEqual(next_idx, current_train_samples + num_added)
        
        expected_put_calls = [
            call(str(5).encode(), pickle.dumps((b"imgA1_data", b"charA"))),
            call(str(6).encode(), pickle.dumps((b"imgB1_data", b"charB"))),
            call(str(7).encode(), pickle.dumps((b"imgA2_data", b"charA"))),
        ]
        # Check calls on the transaction object obtained from train_env.begin().__enter__()
        mock_train_env.begin.return_value.__enter__.return_value.put.assert_has_calls(expected_put_calls, any_order=False)

    def test_update_train_db_metadata(self):
        mock_train_env, mock_train_txn, _ = create_mock_lmdb_env()
        original_meta = {"num_samples": 10, "character_counts": {"a": 5, "b": 5}, "split": "train"}
        num_added = 5
        added_chars = {"c": 3, "a": 2}
        font_db_path = "/path/to/font.lmdb"
        final_char_counts = {"a": 7, "b": 5, "c": 3}
        final_num_samples = 15

        updated_meta = _update_train_db_metadata(
            mock_train_env, original_meta, num_added, added_chars, 
            font_db_path, final_char_counts, final_num_samples
        )

        self.assertEqual(updated_meta['num_samples'], 15)
        self.assertEqual(updated_meta['character_counts'], final_char_counts)
        self.assertIn('merged_font_data_stats', updated_meta)
        self.assertEqual(updated_meta['merged_font_data_stats']['num_samples_added'], num_added)
        self.assertEqual(updated_meta['merged_font_data_stats']['source_path'], font_db_path)
        
        # Check that the metadata was put to the db
        # The call is to the txn object obtained from train_env.begin().__enter__()
        put_call_args = mock_train_env.begin.return_value.__enter__.return_value.put.call_args[0]
        self.assertEqual(put_call_args[0], b'__metadata__')
        self.assertEqual(json.loads(put_call_args[1].decode()), updated_meta)


class TestGentrainMainFunction(unittest.TestCase):

    @patch('src.gentrain.parse_args')
    @patch('os.path.exists')
    @patch('lmdb.open') # Mock lmdb.open used by helper and main
    @patch('src.gentrain._read_lmdb_metadata_and_info')
    @patch('src.gentrain._calculate_target_map_size')
    @patch('src.gentrain._transfer_data_and_update_stats')
    @patch('src.gentrain._update_train_db_metadata')
    @patch('os.makedirs') # Mock os.makedirs for new train DB case
    def test_merge_font_data_to_train_success_flow(self, mock_makedirs, mock_update_meta, mock_transfer, 
                                                 mock_calc_map_size, mock_read_meta, mock_lmdb_open, 
                                                 mock_os_exists, mock_parse_args):
        # --- Setup Mocks ---
        mock_parse_args.return_value = MagicMock(font_db_path="font.lmdb", train_db_path="train.lmdb")
        mock_os_exists.side_effect = lambda path: True # Both DBs exist

        # Mock return for _read_lmdb_metadata_and_info
        # (metadata, db_info, num_samples, map_size, map_used)
        font_meta_fixture = {'total_records': 100}
        train_meta_fixture = {'num_samples': 50, 'character_counts': {'a': 50}}
        mock_read_meta.side_effect = [
            (font_meta_fixture, {'map_size': 10*1024*1024, 'map_used': 5*1024*1024}, 100, 10*1024*1024, 5*1024*1024), # Font DB read
            (train_meta_fixture, {'map_size': 20*1024*1024, 'map_used': 10*1024*1024}, 50, 20*1024*1024, 10*1024*1024)  # Train DB read
        ]

        mock_calc_map_size.return_value = 30 * 1024 * 1024 # Calculated map size

        # Mock lmdb.open calls in the main function (not helpers which are already using a patched lmdb.open)
        mock_font_env_main, _, _ = create_mock_lmdb_env()
        mock_train_env_main, _, _ = create_mock_lmdb_env()
        mock_lmdb_open.side_effect = [mock_font_env_main, mock_train_env_main] # For main func open calls

        # Mock return for _transfer_data_and_update_stats
        # (num_added, added_chars_summary, final_char_counts, final_total_samples_count)
        mock_transfer.return_value = (100, {'b': 100}, {'a': 50, 'b': 100}, 150)

        # --- Execute --- 
        merge_font_data_to_train("font.lmdb", "train.lmdb")

        # --- Assertions ---
        mock_read_meta.assert_any_call("font.lmdb", "FontDB")
        mock_read_meta.assert_any_call("train.lmdb", "TrainDB")
        
        mock_calc_map_size.assert_called_once_with(20*1024*1024, 10*1024*1024, 50, 100, None)
        
        expected_lmdb_calls = [
            call("font.lmdb", readonly=True, lock=False, readahead=False, meminit=False),
            call("train.lmdb", map_size=30 * 1024 * 1024, writemap=True)
        ]
        mock_lmdb_open.assert_has_calls(expected_lmdb_calls)

        mock_transfer.assert_called_once_with(
            mock_font_env_main, mock_train_env_main, 50, {'a': 50}, font_total_items_for_tqdm=100
        )
        mock_update_meta.assert_called_once_with(
            mock_train_env_main, train_meta_fixture, 100, {'b': 100}, "font.lmdb", {'a': 50, 'b': 100}, 150
        )
        self.assertTrue(mock_font_env_main.close.called)
        self.assertTrue(mock_train_env_main.close.called)

    @patch('src.gentrain.parse_args')
    @patch('os.path.exists')
    @patch('lmdb.open')
    @patch('src.gentrain.logger.error') # To check if MapFullError is logged
    def test_merge_font_data_to_train_map_full_error(self, mock_log_error, mock_lmdb_open, mock_os_exists, mock_parse_args):
        mock_parse_args.return_value = MagicMock(font_db_path="font.lmdb", train_db_path="train.lmdb")
        mock_os_exists.return_value = True
        
        # Simulate lmdb.open raising MapFullError when opening train_env for writing
        mock_font_env, _, _ = create_mock_lmdb_env()
        def lmdb_open_side_effect(path, **kwargs):
            if path == "font.lmdb": return mock_font_env
            if path == "train.lmdb" and kwargs.get('map_size'): # This is the train_env open
                raise lmdb.MapFullError("Map is full")
            return MagicMock() # Default for other calls if any
        mock_lmdb_open.side_effect = lmdb_open_side_effect

        # Need to mock _read_lmdb_metadata_and_info as it also calls lmdb.open
        with patch('src.gentrain._read_lmdb_metadata_and_info') as mock_read_meta_helper:
            mock_read_meta_helper.side_effect = [
                ({'total_records': 100}, {}, 100, 10*1024*1024, 5*1024*1024), # Font DB
                ({}, {}, 0, 1*1024*1024*1024, 0) # Train DB (empty)
            ]
            # And _calculate_target_map_size to provide a map_size
            with patch('src.gentrain._calculate_target_map_size', return_value=10*1024*1024) as mock_calc_map:
                 merge_font_data_to_train("font.lmdb", "train.lmdb")
        
        # Check if the error related to MapFullError was logged
        found_log = False
        for call_item in mock_log_error.call_args_list:
            args, _ = call_item
            if args and "LMDB MapFullError" in args[0]:
                found_log = True
                break
        self.assertTrue(found_log, "LMDB MapFullError log not found")
        self.assertTrue(mock_font_env.close.called) # Ensure font_env is closed

    @patch('src.gentrain.parse_args')
    @patch('os.path.exists', return_value=False) # Font DB does not exist
    @patch('src.gentrain.logger.error')
    def test_merge_font_data_to_train_font_db_not_exist(self, mock_log_error, mock_os_exists, mock_parse_args):
        mock_parse_args.return_value = MagicMock(font_db_path="font.lmdb", train_db_path="train.lmdb")
        merge_font_data_to_train("font.lmdb", "train.lmdb")
        
        # Corrected assertion for logger.error call
        found_log = False
        for call_item in mock_log_error.call_args_list:
            args, _ = call_item
            if args and "Font LMDB database not found: font.lmdb" in args[0]:
                found_log = True
                break
        self.assertTrue(found_log, "Font DB not found log not found")

    @patch('src.gentrain.parse_args')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('lmdb.open') # Mock lmdb.open used by helper and main
    @patch('src.gentrain._read_lmdb_metadata_and_info')
    @patch('src.gentrain._calculate_target_map_size')
    @patch('src.gentrain._transfer_data_and_update_stats')
    @patch('src.gentrain._update_train_db_metadata')
    def test_merge_font_data_to_train_new_train_db(self, mock_update_meta, mock_transfer, 
                                                mock_calc_map_size, mock_read_meta_helper, mock_lmdb_open_main, 
                                                mock_makedirs, mock_os_exists, mock_parse_args):
        mock_parse_args.return_value = MagicMock(font_db_path="font.lmdb", train_db_path="new/train.lmdb")
        
        # os.path.exists: font.lmdb exists, new/train.lmdb does not, new/ does not
        mock_os_exists.side_effect = lambda p: True if p == "font.lmdb" else False

        # _read_lmdb_metadata_and_info: font exists, train does not (will be called with non_existent path)
        mock_read_meta_helper.side_effect = [
            ({'total_records': 10}, {}, 10, 10*1024*1024, 5*1024*1024), # Font DB
            ({}, {}, 0, 1*1024*1024*1024, 0) # For train DB (as it doesn't exist, helper returns defaults)
        ]
        mock_calc_map_size.return_value = 100*1024*1024
        mock_transfer.return_value = (10, {'c': 10}, {'c': 10}, 10) # num_added, added_summary, final_counts, next_idx

        mock_font_env_main, _, _ = create_mock_lmdb_env()
        mock_train_env_main, _, _ = create_mock_lmdb_env()
        mock_lmdb_open_main.side_effect = [mock_font_env_main, mock_train_env_main]

        merge_font_data_to_train("font.lmdb", "new/train.lmdb")

        mock_makedirs.assert_called_once_with("new", exist_ok=True)
        mock_update_meta.assert_called_once() # Check it proceeds to update metadata
        self.assertTrue(mock_font_env_main.close.called)
        self.assertTrue(mock_train_env_main.close.called)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 