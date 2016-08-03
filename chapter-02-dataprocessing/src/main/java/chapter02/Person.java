package chapter02;

public class Person {
    private final String name;
    private final String email;
    private final String country;
    private final int salary;
    private final int experience;

    public Person(String name, String email, String country, int salary, int experience) {
        this.name = name;
        this.email = email;
        this.country = country;
        this.salary = salary;
        this.experience = experience;
    }

    public String getName() {
        return name;
    }

    public String getEmail() {
        return email;
    }

    public String getCountry() {
        return country;
    }

    public int getSalary() {
        return salary;
    }

    public int getExperience() {
        return experience;
    }

    @Override
    public String toString() {
        return "Person [name=" + name + ", email=" + email + ", country=" + country + ", salary=" + salary
                + ", experience=" + experience + "]";
    }

}