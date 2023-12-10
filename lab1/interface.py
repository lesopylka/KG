import customtkinter
import main

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("320x240")
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label1 = customtkinter.CTkLabel(master=frame, text="parametr a value")
label1.pack(pady=30, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Enter parametr value")
entry1.pack(pady=10, padx=10)

def DoIt():
    value = float(entry1.get())
    main.doit(value)

button = customtkinter.CTkButton(master=frame, text="Do it", command=DoIt)
button.pack(pady=10, padx=10)

root.mainloop()