import subprocess

def run_permission_script(app_name):
    output = ""
    if app_name.lower() in ["whatsapp", "snapchat", "dream11"]:
        # Run Permissio.py with the given app_name
        command = ["python", "Permissio.py", app_name]
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout
    else:
        # Run p.py with the given app_name
        command = ["python", "p.py", app_name]
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout

    return output

if __name__ == "__main__":
    app_name = input("Enter the name of the app: ").strip()
    output = run_permission_script(app_name)
    print(output)
    
    # List of permissions
    permissions = [
        "android.permission.READ_SMS",
        "android.permission.MANAGE_EXTERNAL_STORAGE",
        "android.permission.ACCESS_BLUETOOTH_SHARE",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.POST_NOTIFICATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.intent.category.MASTER_CLEAR.permission.C2D_MESSAGE",
        "android.os.cts.permission.TEST_GRANTED",
        "android.permission.ACCESS_INPUT_FLINGER",
        "android.permission.ACCESS_KEYGUARD_SECURE_STORAGE",
        "android.permission.ACCESS_BLUETOOTH_SHARE",
        "android.permission.ACCESS_CACHE_FILESYSTEM",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.permission.ACCESS_WIFI_STATE",
        "android.permission.ACCESS_WIMAX_STATE",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.intent.category.MASTER_CLEAR.permission.C2D_MESSAGE",
        "android.os.cts.permission.TEST_GRANTED",
        "android.permission.AUTHENTICATE_ACCOUNTS",
        "android.permission.BACKUP",
        "android.permission.BATTERY_STATS",
        "android.permission.ACCESS_CACHE_FILESYSTEM",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.INTERNET",
        "android.permission.ACCESS_NOTIFICATIONS",
    ]

    # Create a list of 0's and 1's
    binary_list = ['1' if permission in output else '0' for permission in permissions]

    # Join the list into a comma-separated string
    binary_string = ';'.join(binary_list)
    
    print(binary_string)
# After generating the binary_string
with open("binary_permissions.txt", "w") as file:
    file.write(binary_string)

