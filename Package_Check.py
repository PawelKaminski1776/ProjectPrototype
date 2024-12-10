import pkg_resources

if __name__ == "__main__":
    installed_packages = pkg_resources.working_set
    for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
        print(f"{package.project_name}=={package.version}")
