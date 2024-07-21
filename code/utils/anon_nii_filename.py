import os
import argparse

def format_sub_id(sub_id):
    """
    Format the sub id to ensure it's in the format 'sub-xx'.
    
    Args:
    - sub_id (str): Subject ID, which can be in various formats like '1', '01', or 'sub-01'.
    
    Returns:
    - str: Formatted subject ID.
    """
    if sub_id.startswith('sub-'):
        sub_number = int(sub_id.split('-')[1])
    else:
        sub_number = int(sub_id)
    return f'sub-{sub_number:02d}'

def get_relative_path(path):
    """
    Convert an absolute path to a relative path based on the current working directory.
    
    Args:
    - path (str): The absolute path to be converted.
    
    Returns:
    - str: The relative path.
    """
    return os.path.relpath(path)

def rename_files_in_directory(directory, confirm=True, dry_run=False):
    """
    Collect and optionally rename files within a specified directory based on specific conditions.
    
    Args:
    - directory (str): Path to the directory where files will be scanned and possibly renamed.
    - confirm (bool): If True, confirmation is required before renaming files.
    - dry_run (bool): If True, no files will be actually renamed; only planned changes will be shown.
    
    Returns:
    - List[Tuple[str, str]]: A list of tuples containing old and new paths for files to be renamed.
    """
    changes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '_WIP_' in file:
                original_path = os.path.join(root, file)
                sub_id = os.path.basename(os.path.dirname(root))
                new_name = f"{sub_id}_WIP_{file.split('_WIP_', 1)[1]}"
                new_path = os.path.join(root, new_name)
                changes.append((original_path, new_path))

    if changes:
        print("Planned changes:")
        for old, new in changes:
            print(f"{get_relative_path(old)} --> {get_relative_path(new)}")

    if not dry_run and confirm:
        response = input("Proceed with these changes? (y/n): ")
        if response.lower() == 'y':
            for old, new in changes:
                os.rename(old, new)

    return changes

def main():
    """
    Main function to handle command line arguments and control the flow of the script.
    """
    parser = argparse.ArgumentParser(description='Rename files based on specific criteria.')
    parser.add_argument('--level', choices=['group', 'participant'], required=True, help='Processing level, group or participant')
    parser.add_argument('--confirm', type=bool, default=True, help='Confirm before renaming (default: True)')
    parser.add_argument('--dry_run', action='store_true', help='Only display planned changes without renaming')
    parser.add_argument('--sub', nargs='+', help='Specific sub IDs to process')

    args = parser.parse_args()

    base_directory = os.getcwd()  # Use the current working directory
    if args.sub:
        sub_ids = [format_sub_id(sub) for sub in args.sub]
    else:
        sub_ids = sorted([
            dir_name for dir_name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, dir_name)) and dir_name.startswith('sub-')
        ], key=lambda x: int(x.split('-')[1]))

    if args.level == 'group':
        all_changes = []
        for sub_id in sub_ids:
            directory = os.path.join(base_directory, sub_id)
            changes = rename_files_in_directory(directory, confirm=False, dry_run=args.dry_run)
            all_changes.extend(changes)
        if not args.dry_run and args.confirm and all_changes:
            response = input("Proceed with all these changes for the group? (y/n): ")
            if response.lower() == 'y':
                for old, new in all_changes:
                    os.rename(old, new)
    elif args.level == 'participant':
        for sub_id in sub_ids:
            directory = os.path.join(base_directory, sub_id)
            rename_files_in_directory(directory, confirm=args.confirm, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
