from project_questions import ProjectQuestions

if __name__ == "__main__":
    question_number = 1

    vo_data = {}
    vo_data['dir'] = r"D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\HW_3\Part3_VO\dataset"
    vo_data['sequence'] = 2

    if question_number == 1 or question_number == 2 or question_number == 3:
        project = ProjectQuestions(question_number, vo_data)
    else:
        print("Invalid question number")
        exit()

    project.run()
