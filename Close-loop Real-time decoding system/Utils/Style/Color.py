# -*- coding: UTF-8 -*-

"""
@Project：TMR_Block
@File：Color.py

@Organization: Beijing Normal University
@Author：Peiyang Lin
@Email: langaLinn@gmail.com
@Computer：Q58 Langa
@Date ：2024/3/22 16:30

@IDE：PyCharm 
"""

# TODO: Start of the whole program
from colorama import init, Fore, Back, Style

init(autoreset=True)


class Color:
    @staticmethod
    def black(s):
        return Fore.BLACK + s + Fore.RESET

    @staticmethod
    def red(s):
        return Fore.RED + s + Fore.RESET

    @staticmethod
    def green(s):
        return Fore.GREEN + s + Fore.RESET

    @staticmethod
    def yellow(s):
        return Fore.YELLOW + s + Fore.RESET

    @staticmethod
    def blue(s):
        return Fore.BLUE + s + Fore.RESET

    @staticmethod
    def magenta(s):
        return Fore.MAGENTA + s + Fore.RESET

    @staticmethod
    def cyan(s):
        return Fore.CYAN + s + Fore.RESET

    @staticmethod
    def white(s):
        return Fore.WHITE + s + Fore.RESET

    @staticmethod
    def lightBlack(s):
        return Fore.LIGHTBLACK_EX + s + Fore.RESET

    @staticmethod
    def lightRed(s):
        return Fore.LIGHTRED_EX + s + Fore.RESET

    @staticmethod
    def lightGreen(s):
        return Fore.LIGHTGREEN_EX + s + Fore.RESET

    @staticmethod
    def lightYellow(s):
        return Fore.LIGHTYELLOW_EX + s + Fore.RESET

    @staticmethod
    def lightBlue(s):
        return Fore.LIGHTBLUE_EX + s + Fore.RESET

    @staticmethod
    def lightMagenta(s):
        return Fore.LIGHTMAGENTA_EX + s + Fore.RESET

    @staticmethod
    def lightCyan(s):
        return Fore.LIGHTCYAN_EX + s + Fore.RESET

    @staticmethod
    def lightWhite(s):
        return Fore.LIGHTWHITE_EX + s + Fore.RESET
