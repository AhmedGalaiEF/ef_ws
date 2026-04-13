# Install init file
python_install_on_site(${PROJECT_NAME} __init__.py)
python_install_on_site(${PROJECT_NAME} loader.py)
python_install_on_site(${PROJECT_NAME} go2_joint_trot.py)
python_install_on_site(${PROJECT_NAME} go2_joint_bounce.py)
python_install_on_site(${PROJECT_NAME} go2_joint_step.py)
python_install_on_site(${PROJECT_NAME} go2_joint_sit_stand.py)
python_install_on_site(${PROJECT_NAME} go2_joint_sway.py)
python_install_on_site(${PROJECT_NAME} go2_joint_bow.py)

# Populate the path file and install it manually
configure_file(${PROJECT_NAME}/path.py.in ${PROJECT_NAME}/path.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/path.py
        DESTINATION "${PYTHON_SITELIB}/${PROJECT_NAME}")
