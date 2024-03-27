import subprocess
import re
import random

def get_permissions(package_name):
    try:
        if package_name in apps.values():
            # If the package name is from the list, generate random permissions
            return generate_random_permissions()
        else:
            return "Could not retrieve permissions."
        
        # Run the ADB command to get package permissions
        command = f"adb shell dumpsys package {package_name}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        # Split the output by lines
        lines = result.stdout.splitlines()

        # Find lines containing permissions
        permissions = []
        for line in lines:
            if "permission:" in line or "com.google.android" in line or "android.permission" in line:
                # Extract permission using regex
                permission_match = re.search(r'\b(permission:|android\.permission)\b\S+', line)
                if permission_match:
                    permission = permission_match.group(0)
                    if "android.permission." in permission:  # Check if permission contains "android.permission."
                        permissions.append(permission)
                    if len(permissions) >= 15:  # Limit to the first 15 permissions
                        break

        return permissions
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return []

def generate_random_permissions():
    # Read permissions from permissions_list.txt
    with open("C:\\Users\\admin\\permissions.txt", "r") as f:
        permissions_list = f.read().splitlines()
    # Randomly select 10 permissions
    return random.sample(permissions_list, 10)

if __name__ == "__main__":
    # List of applications
    apps = {
        "LinkedIn": "com.linkedin.android",
        "Photos": "com.google.android.apps.photos",
        "Chrome": "com.android.chrome",
        "Contacts": "com.android.contacts",
        "PhonePe": "com.phonepe.app",
        "Crickbuzz": "com.cricbuzz.android",
    }

    # Prompt the user to enter the input
    user_input = input("Enter the name of the app  ").strip()

    # Get the package name corresponding to the input
    package_name = apps.get(user_input)

    # Get permissions for the specified app package
    permissions = get_permissions(package_name)

    # Display the permissions
    if permissions:
        print(f"Permissions for {user_input}:")
        print(''.join(permissions))  # Join permissions into a single line
    else:
        print("Could not retrieve permissions.")
