import subprocess
import re

def get_permissions(package_name):
    try:
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

if __name__ == "__main__":
    # Dictionary mapping input names to package names
    apps = {
        "Snapchat": "com.snapchat.android",
        "Whatsapp": "com.whatsapp",
        "Dream11": "com.app.dream11Pro",
        "Pikashow": "com.pikashow.app",
        "5th": "com.example.app",  # Default package name for other inputs
    }

    # Prompt the user to enter the input
    user_input = input("Enter the name of the app  ").strip()

    # Get the package name corresponding to the input
    package_name = apps.get(user_input, apps["5th"])

    # Get permissions for the specified app package
    permissions = get_permissions(package_name)

    # Display the permissions
    if permissions:
        print(f"Permissions for {user_input}:")
        for permission in permissions:
            print(permission)
    else:
        print("Could not retrieve permissions.")
